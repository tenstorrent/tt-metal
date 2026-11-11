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
            # per-user positions + RoPE rows for the engine (attention RQNR / KV append): the PERSISTENT packed aux
            # _vfy_aux_buf (QWEN36_SPEC_HOSTSET, prod default): uint32 ROW_MAJOR [3, W] replicated DRAM, one page per row;
            # row 0 = kvpos bits (int32, -1 = HOLD) at r = u*T + j, row 1 = RoPE table row (position + rope_delta: the
            # M-RoPE delta of image prompts, so NOT kvpos), row 2 = the GDN ctrl page. RoPE tables: rope.decode_rm_tables()
            # bf16 ROW_MAJOR [max_seq_len, rope_dim] (cos, sin), the exact rows prod's in-trace lookup reads.
            aux = getattr(model, "_vfy_aux_buf", None)
            if aux is not None:
                d.update(aux_addr=aux.buffer_address(), aux_w=int(aux.shape[-1]), aux_rows=int(aux.shape[0]))
            try:
                cos_t, sin_t = model.rope.decode_rm_tables()
                d.update(rope_cos_addr=cos_t.buffer_address(), rope_sin_addr=sin_t.buffer_address(),
                         rope_max=int(cos_t.shape[0]), rope_dim=int(cos_t.shape[-1]))
            except Exception:  # noqa: BLE001
                pass
            pt = getattr(model, "_vfy_kvpt_users", None)  # the engine's page table (see per_replay_inputs)
            d.update(page_table=pt, pt_addr=(pt.buffer_address() if pt is not None else 0),
                     pt_B=int(getattr(model, "_vfy_B", 0) or 0), pt_nb=int(getattr(model, "_vfy_nb", 0) or 0))
            out.append(d)
    return out


def gdn_descriptors(model):
    """Per GDN layer: the spec-verify state the engine's GDN role reads / writes, keyed like the role's per-layer table
    (gdn/u1/g2_role.py GdnRole.tl: win_a / win_b / ring / taps / dtb / nea / weight), i.e. exactly the tensors prod's
    gdn_spec_step gets (gdn/tp.py _verify_fullbatch_fused): the window ping-pong pair _verify_win_pair (the kernels pick
    the half by the ctrl parity), the per-token state ring _spec_ring (fp32 [T*B*Nv, Dk, Dv]), and the layer constants
    conv_taps_rows / dt_bias_fp32 / neg_exp_A_fp32 / norm_w_1d. ctrl is ONE page for every layer, staged per replay
    (per_replay 'ctrl'). Also the unfused path's selectors (state_blk_idx / conv_sel) and the durable rec_state /
    conv_states (the prefill -> decode hand-off: seed_spec_state copies them into the ring / window before the loop)."""
    def addr(t):
        return t.buffer_address() if t is not None and hasattr(t, "buffer_address") else 0

    out = []
    sel = dict(state_idx_addr=addr(getattr(model, "_vfy_state_idx", None)),
               conv_sel_addr=addr(getattr(model, "_vfy_conv_sel", None)))
    for i, l in enumerate(model.layers):
        if l.is_full_attention:
            continue
        a = l.attention
        tw = getattr(a, "tw", {}) or {}
        pair = getattr(a, "_verify_win_pair", None) or (getattr(a, "_verify_win_buf", None), None)
        t = dict(win_a=pair[0], win_b=pair[1], ring=getattr(a, "_spec_ring", None), taps=tw.get("conv_taps_rows"),
                 dtb=tw.get("dt_bias_fp32"), nea=tw.get("neg_exp_A_fp32"), weight=tw.get("norm_w_1d"),
                 rec_state=getattr(a, "rec_state", None))
        out.append(dict(layer=i, tensors=t, addrs={k: addr(v) for k, v in t.items()},
                        fused=bool(getattr(a, "_spec_fused", False)), win_par=int(getattr(a, "_spec_win_par", 0)),
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


def _named_tensors(root, prefix, out, depth=5, seen=None):
    """{buffer address: attribute path} for every allocated device ttnn.Tensor reachable from ``root`` through
    attributes / dicts / lists (pybind tensors are not gc-tracked, so gc.get_objects() cannot see them)."""
    import ttnn

    seen = set() if seen is None else seen
    if id(root) in seen or depth < 0:
        return out
    seen.add(id(root))
    if isinstance(root, ttnn.Tensor):
        try:
            if root.storage_type() == ttnn.StorageType.DEVICE and root.is_allocated():
                out.setdefault(root.buffer_address(), prefix)
        except Exception:  # noqa: BLE001
            pass
        return out
    if isinstance(root, dict):
        items = list(root.items())
    elif isinstance(root, (list, tuple)):
        items = list(enumerate(root))
    elif hasattr(root, "__dict__") and type(root).__module__.startswith(("models.", "__main__")):
        items = list(vars(root).items())
    else:
        return out
    for k, v in items[:4096]:
        if isinstance(v, (int, float, str, bytes, bool)) or v is None:
            continue
        _named_tensors(v, f"{prefix}.{k}" if not isinstance(k, int) else f"{prefix}[{k}]", out, depth - 1, seen)
    return out


def l1_dump(tag, prog=None, model=None, limit=80, cores=None):
    """Debug (QWEN36_R16_MK_L1DUMP=1): every live L1 buffer from the device allocator (ttnn reports), lowest address
    first, sized per bank, named by the model / engine attribute that holds it ('?' = op-internal: trace intermediates,
    CCL persistent buffers, ...). ``cores`` (default the owner rect x10-11 y0-7): which sharded buffers sit there."""
    import ttnn
    from loguru import logger

    mesh = model.mesh_device if model is not None else None
    S = getattr(getattr(prog, "program", None), "setup", None) or {}
    names = {}
    if model is not None:
        _named_tensors(model, "model", names)
    _named_tensors(S, "engine", names)
    blocks = getattr(getattr(prog, "program", None), "blocks", None)
    if blocks is not None:
        _named_tensors({k: v["tensor"] for k, v in blocks.blocks.items()}, "argblock", names)
    if mesh is None:
        logger.info(f"[mk-l1] {tag}: no mesh")
        return []
    rep = ttnn._ttnn.reports
    try:
        info = rep.get_device_info(mesh)
        top, cb0 = info.worker_l1_size, info.address_at_first_l1_cb_buffer
    except Exception:  # noqa: BLE001
        top, cb0 = 0, 0
    cores = cores or {(x, y) for x in (10, 11) for y in range(8)}
    on_cores = {}
    try:
        for pg in rep.get_buffer_pages(mesh):
            if (pg.core_x, pg.core_y) in cores:
                on_cores.setdefault(pg.address, set()).add((pg.core_x, pg.core_y))
    except Exception as e:  # noqa: BLE001
        logger.info(f"[mk-l1] get_buffer_pages failed: {e}")
    rows = []
    for b in rep.get_buffers(mesh):
        if b.buffer_type != ttnn.BufferType.L1:
            continue
        lay = str(b.buffer_layout).split(".")[-1]
        rows.append((b.address, b.max_size_per_bank, lay, names.get(b.address, "?"), len(on_cores.get(b.address, ()))))
    rows = sorted(set(rows))
    low = min((r[0] for r in rows if r[4]), default=None)
    tot = sum(r[1] for r in rows if r[4])
    logger.info(f"[mk-l1] {tag}: {len(rows)} live L1 buffers; L1 top {top} CB base {cb0}; on the owner cores: "
                f"{sum(1 for r in rows if r[4])} buffers, {tot} B per core, lowest {low}")
    for r in rows[:limit]:
        logger.info(f"[mk-l1]   addr {r[0]:>8} +{r[1]:>7} {r[2]:>15} owners {r[4]:>2} {r[3][:160]}")
    return rows


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
        dump = os.environ.get("QWEN36_R16_MK_L1DUMP") == "1"
        if dump:
            l1_dump("before engine setup", None, self.model)
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
            if dump:
                l1_dump(f"after engine setup bucket {b}", prog, self.model)
        if not self.programs:
            self.reason = f"no bucket in scope ({self.skipped})"
            return {"ready": False, "reason": self.reason}
        self.ready = True
        self.replaces_layers = all(p.replaces_layers for p in self.programs.values())
        return {"ready": True, "buckets": sorted(self.programs), "replaces_layers": self.replaces_layers,
                "kind": type(next(iter(self.programs.values()))).__name__, "out_of_scope": sorted(self.skipped),
                "info": {b: getattr(p, "info", None) for b, p in self.programs.items()}}

    @staticmethod
    def _rows(row_positions, inactive, T):
        rows = [int(p) for p in row_positions]
        if inactive is not None:
            for u, off in enumerate(inactive):
                if off:
                    rows[u * T:(u + 1) * T] = [-1] * T
        return rows

    def post_enqueue(self, row_positions, inactive, T):
        """Right after replay k is enqueued: precompute every program's host tables for replay k+1 from the upper-bound
        positions (each live row + T; the tables are hints), so step k+1 only pays the H2D copy."""
        rows = [p + T if p >= 0 else -1 for p in self._rows(row_positions, inactive, T)]
        for prog in self.programs.values():
            pre = getattr(getattr(prog, "program", None), "precompute", None)
            if pre is not None:
                pre(rows)

    def stage_host(self, row_positions, inactive, T):
        """Before each verify replay (model.verify_stage_and_enqueue): every engine program's host-staged tables from the
        step's row positions (r = u*T + j; -1 for inactive users). No-op for programs without host tables."""
        rows = self._rows(row_positions, inactive, T)
        for prog in self.programs.values():
            st = getattr(getattr(prog, "program", None), "stage", None)
            if st is not None:
                st(rows)

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
            l1_dump(f"engine launch bucket {n_users}", prog, self.model)
        prog.launch(x_in, x_out, per_replay_inputs(self.model, in_body))
        ttnn.deallocate(x_in)
        return ttnn.to_memory_config(x_out, mc)
