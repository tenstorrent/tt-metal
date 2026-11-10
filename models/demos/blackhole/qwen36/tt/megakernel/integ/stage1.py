"""Megakernel integration stage1 host plumbing (QWEN36_R16_MEGAKERNEL=stage1): replace the verify layer loop with the
engine's one-generic_op-per-pass program, per DESIGN_integration.md §1-§5 and the engine entry-point contract proposed
to the megakernel agent:

    build_engine(mesh, layer_types, n_users, weights_mk, kv_desc, gdn_desc, fabric_ctx, T=4) -> EngineProgram
    EngineProgram.alloc_x_out(); EngineProgram.launch(x_in, x_out, per_replay); EngineProgram.replaces_layers
(engine/build.py; spec v0.9.3 records the agreed contract.)

Everything here is host-side and engine-agnostic. It gathers the model's tensors (KV caches, GDN state, per-replay
verify inputs), owns the persistent x_out residual per bucket, and calls the engine builder once per bucket before
capture. Until an engine builder is importable, ``Stage1Verify.ready`` is False with a reason, and the model keeps
the op-by-op layer loop.
"""

import importlib
import os

ENGINE_MODULE = os.environ.get("QWEN36_R16_MK_ENGINE", "models.demos.blackhole.qwen36.tt.megakernel.engine")


def _engine_builder():
    try:
        mod = importlib.import_module(ENGINE_MODULE)
    except ImportError as e:
        return None, f"engine module {ENGINE_MODULE} not importable ({e})"
    fn = getattr(mod, "build_engine", None)
    return (fn, None) if callable(fn) else (None, f"{ENGINE_MODULE} has no build_engine()")


def kv_descriptors(model):
    """Per full-attention layer, the KV cache the verify actually reads.

    vLLM / prod (``use_paged``): two tensors, paged_k and paged_v, each [num_blocks, n_kv_heads, block_size, head_dim]
    TILE, BFP8, replicated on every device. Tile page of (block p, head h, position tile r, dim tile c) =
    ((p * n_kv_heads + h) * (block_size // 32) + r) * (head_dim // 32) + c. So K and V are SEPARATE base addresses, unlike
    the bench's interleaved KVCacheDesc (one tensor, 8 K then 8 V tiles per block); the engine reader takes two bases.
    Demo fallback (no vLLM cache): k_caches / v_caches lists, one [B, 1, max_seq_len, head_dim] bf16 tensor per KV head
    (not paged; stage1 does not target it)."""
    out = []
    for i, l in enumerate(model.layers):
        if l.is_full_attention:
            a = l.attention
            paged = bool(getattr(a, "use_paged", False))
            k, v = (a.paged_k, a.paged_v) if paged else (getattr(a, "k_caches", None), getattr(a, "v_caches", None))
            d = dict(layer=i, paged=paged, k=k, v=v)
            if paged and k is not None:
                nb, nh, bs, hd = (int(x) for x in k.shape)
                d.update(num_blocks=nb, n_kv_heads=nh, block_size=bs, head_dim=hd, dtype=str(k.dtype),
                         k_addr=k.buffer_address(), v_addr=v.buffer_address())
            pt = getattr(model, "_vfy_kvpt_users", None)  # the engine's page table (see per_replay_inputs)
            d.update(page_table=pt, pt_addr=(pt.buffer_address() if pt is not None else 0),
                     pt_B=int(getattr(model, "_vfy_B", 0) or 0), pt_nb=int(getattr(model, "_vfy_nb", 0) or 0))
            out.append(d)
    return out


def gdn_descriptors(model):
    """Per GDN layer: the spec-verify state the engine's GDN phases read / write (the per-token state ring and the conv
    window of prepare_spec_verify, rec_state / conv_states), plus the shared selectors (state_blk_idx / conv_sel)."""
    def addr(t):
        return t.buffer_address() if t is not None and hasattr(t, "buffer_address") else 0

    out = []
    sel = dict(state_idx_addr=addr(getattr(model, "_vfy_state_idx", None)),
               conv_sel_addr=addr(getattr(model, "_vfy_conv_sel", None)),
               ctrl_addr=addr(getattr(model, "_vfy_ctrl", None)))
    for i, l in enumerate(model.layers):
        if not l.is_full_attention:
            a = l.attention
            win = getattr(a, "_verify_win_buf", None)
            out.append(dict(layer=i, ring=getattr(a, "_spec_ring", None), ring_addr=addr(getattr(a, "_spec_ring", None)),
                            win_addr=addr(win), rec_state_addr=addr(getattr(a, "rec_state", None)),
                            conv_addrs=[addr(c) for c in (getattr(a, "conv_states", None) or ())], **sel))
    return out


def per_replay_inputs(model, in_body):
    """The verify inputs of this replay. ``in_body`` holds the tensors the trace body derived (kvpos / ctrl / cos / sin
    from the packed aux under QWEN36_SPEC_HOSTSET, the bucket-sliced page tables); the persistent GDN selectors come
    from the model. Only addresses go into runtime args."""
    d = dict(state_blk_idx=getattr(model, "_vfy_state_idx", None), conv_sel=getattr(model, "_vfy_conv_sel", None))
    # The engine's page table: the PERSISTENT per-user table _vfy_kvpt_users, int32 [B, nb] ROW_MAJOR, DRAM interleaved
    # (one row = one page of nb*4 B), replicated on every device; row u = verify-batch user slot u (the token / kvpos
    # row order), entry b = physical block index into paged_k / paged_v dim 0, zero-padded past the user's blocks
    # (bound reads by kvpos). Fixed address (allocated in prepare_verify_trace before warmup), rewritten in place by
    # verify_traced(page_tables=...) before execute_trace (copy skipped when unchanged). A bucket of n_users < B reads
    # rows 0..n_users-1 of this same buffer: use it, not the in-body pt_users (a per-trace ttnn.slice copy).
    pt = getattr(model, "_vfy_kvpt_users", None)
    d.update(page_table=pt, page_table_addr=(pt.buffer_address() if pt is not None else None),
             page_table_B=getattr(model, "_vfy_B", None), page_table_nb=getattr(model, "_vfy_nb", None))
    d.update(in_body)
    return d


def l1_dump(tag, prog=None, limit=120):
    """Debug (QWEN36_R16_MK_L1DUMP=1): every live device L1 tensor (address, shape, layout, shard grid), lowest address
    first, marking the engine's own setup tensors -- to see what sits inside the engine's static CB region."""
    import gc

    import ttnn
    from loguru import logger

    eng = {}
    S = getattr(getattr(prog, "program", None), "setup", None) or {}
    for k, v in S.items():
        for t in (v if isinstance(v, list) else [v]):
            if isinstance(t, ttnn.Tensor):
                eng[id(t)] = k
    rows = []
    for o in gc.get_objects():
        try:
            if isinstance(o, ttnn.Tensor) and o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                mc = o.memory_config()
                if mc.buffer_type == ttnn.BufferType.L1:
                    ss = mc.shard_spec
                    sg = f"{ss.grid} shard {list(ss.shape)}" if ss is not None else "interleaved(all cores)"
                    try:
                        vol = o.volume() if callable(getattr(o, "volume", None)) else int(__import__("math").prod(o.padded_shape))
                    except Exception:  # noqa: BLE001
                        vol = 0
                    rows.append((o.buffer_address(), tuple(o.shape), str(o.dtype).split(".")[-1], str(mc.memory_layout).split(".")[-1],
                                 sg, eng.get(id(o), "model"), vol))
        except Exception:  # noqa: BLE001
            pass
    rows.sort()
    logger.info(f"[mk-l1] {tag}: {len(rows)} live L1 tensors (engine-owned {sum(r[5] != 'model' for r in rows)})")
    for r in rows[:limit]:
        logger.info(f"[mk-l1]   addr {r[0]:>8} {r[5]:>10} {r[2]:>9} {r[3]:>15} shape {r[1]} vol {r[6]} {r[4][:200]}")


class Stage1Verify:
    def __init__(self, model):
        self.model = model
        self.layer_types = ["attention" if l.is_full_attention else "gdn" for l in model.layers]
        self.builder, self.reason = _engine_builder()
        self.weights_mk = None
        self.programs = {}  # n_users -> EngineProgram
        self.x_out = {}  # n_users -> persistent residual out
        self.ready = False
        self.replaces_layers = False
        self.weights_reason = None
        self.skipped = {}  # bucket -> reason (out of the engine's scope: op-by-op)

    def prepare(self, buckets, T):
        """Before capture: repack weights once, then build each bucket's engine program and its persistent x_out."""
        if self.builder is None:
            return {"ready": False, "reason": self.reason}
        from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk

        if self.weights_mk is None:
            try:
                mod = importlib.import_module(ENGINE_MODULE)
                lay = getattr(mod, "weight_layouts", lambda m: None)(self.model.mesh_device)
                self.weights_mk = weights_mk.repack(self.model, layouts=lay)
            except NotImplementedError as e:  # no layouts = a stand-in engine; a real engine raises on None
                self.weights_mk, self.weights_reason = None, str(e)
        kv, gdn = kv_descriptors(self.model), gdn_descriptors(self.model)
        for b in buckets:
            try:
                prog = self.builder(self.model.mesh_device, self.layer_types, b, self.weights_mk, kv, gdn, None, T=T)
            except Exception as e:  # noqa: BLE001 -- engine.build.OutOfScope: this bucket stays op-by-op
                if type(e).__name__ == "OutOfScope":
                    self.skipped[b] = str(e)
                    continue
                if not isinstance(e, (NotImplementedError, ValueError)):
                    raise
                self.programs.clear()
                self.reason = f"bucket {b}: {e}"
                return {"ready": False, "reason": self.reason}
            alloc = getattr(prog, "alloc_x_out", None)
            if alloc is None:  # contract: the engine owns the residual layout (RES40 replicated), so it allocates x_out
                self.programs.clear()
                return {"ready": False, "reason": "EngineProgram has no alloc_x_out()"}
            self.programs[b], self.x_out[b] = prog, alloc()
        if not self.programs:
            self.reason = f"no bucket in scope ({self.skipped})"
            return {"ready": False, "reason": self.reason}
        self.ready = True
        self.replaces_layers = all(p.replaces_layers for p in self.programs.values())
        return {"ready": True, "buckets": sorted(self.programs), "replaces_layers": self.replaces_layers,
                "kind": type(next(iter(self.programs.values()))).__name__, "out_of_scope": sorted(self.skipped),
                "info": {b: getattr(p, "info", None) for b, p in self.programs.items()}}

    def forward(self, x, n_users, T, in_body):
        """Inside the traced verify body, on the replicated RES40 residual ``x`` [1,1,rows,5120] (consumed). Runs the
        engine (x_in = DRAM copy of x) into the persistent x_out and returns a FRESH tensor in x's memory config (never
        x_out itself: the caller deallocates what it gets back)."""
        import ttnn

        prog = self.programs[n_users]
        x_out = self.x_out[n_users]
        mc = x.memory_config()
        x_in = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if x_in is not x:
            ttnn.deallocate(x)
        if os.environ.get("QWEN36_R16_MK_L1DUMP") == "1":
            l1_dump(f"engine launch bucket {n_users}", prog)
        prog.launch(x_in, x_out, per_replay_inputs(self.model, in_body))
        ttnn.deallocate(x_in)
        return ttnn.to_memory_config(x_out, mc)
