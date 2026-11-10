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
            out.append(d)
    return out


def gdn_descriptors(model):
    """Per GDN layer: (layer index, recurrent state, conv states). Used once the real GDN phases replace stand-ins."""
    out = []
    for i, l in enumerate(model.layers):
        if not l.is_full_attention:
            a = l.attention
            out.append(dict(layer=i, rec_state=getattr(a, "rec_state", None), conv_states=getattr(a, "conv_states", None)))
    return out


def per_replay_inputs(model, in_body):
    """The verify inputs of this replay. ``in_body`` holds the tensors the trace body derived (kvpos / ctrl / cos / sin
    from the packed aux under QWEN36_SPEC_HOSTSET, the bucket-sliced page tables); the persistent GDN selectors come
    from the model. Only addresses go into runtime args."""
    d = dict(state_blk_idx=getattr(model, "_vfy_state_idx", None), conv_sel=getattr(model, "_vfy_conv_sel", None))
    d.update(in_body)
    return d


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

    def prepare(self, buckets, T):
        """Before capture: repack weights once, then build each bucket's engine program and its persistent x_out."""
        if self.builder is None:
            return {"ready": False, "reason": self.reason}
        from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk

        if self.weights_mk is None:
            try:
                self.weights_mk = weights_mk.repack(self.model)
            except NotImplementedError as e:  # stand-in engines take no weights; a real engine raises on None
                self.weights_mk, self.weights_reason = None, str(e)
        kv, gdn = kv_descriptors(self.model), gdn_descriptors(self.model)
        for b in buckets:
            try:
                prog = self.builder(self.model.mesh_device, self.layer_types, b, self.weights_mk, kv, gdn, None, T=T)
            except (NotImplementedError, ValueError) as e:
                self.programs.clear()
                self.reason = f"bucket {b}: {e}"
                return {"ready": False, "reason": self.reason}
            alloc = getattr(prog, "alloc_x_out", None)
            if alloc is None:  # contract: the engine owns the residual layout (RES40 replicated), so it allocates x_out
                self.programs.clear()
                return {"ready": False, "reason": "EngineProgram has no alloc_x_out()"}
            self.programs[b], self.x_out[b] = prog, alloc()
        self.ready = True
        self.replaces_layers = all(p.replaces_layers for p in self.programs.values())
        return {"ready": True, "buckets": sorted(self.programs), "replaces_layers": self.replaces_layers,
                "kind": type(next(iter(self.programs.values()))).__name__}

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
        prog.launch(x_in, x_out, per_replay_inputs(self.model, in_body))
        ttnn.deallocate(x_in)
        return ttnn.to_memory_config(x_out, mc)
