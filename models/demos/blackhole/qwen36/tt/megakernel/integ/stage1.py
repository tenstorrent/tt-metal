"""Megakernel integration stage1 host plumbing (QWEN36_R16_MEGAKERNEL=stage1): replace the verify layer loop with the
engine's one-generic_op-per-pass program, per DESIGN_integration.md §1-§5 and the engine entry-point contract proposed
to the megakernel agent:

    build_engine(mesh, layer_types, n_users, weights_mk, kv_desc, gdn_desc, fabric_ctx) -> EngineProgram
    EngineProgram.io(x_in, x_out, per_replay) -> list of io tensors;  EngineProgram.mesh_program -> MeshProgramDescriptor

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
    """Per full-attention layer: (layer index, k cache tensors, v cache tensors) as the TPAttention module holds them
    (paged BFP8, one tensor per local KV head)."""
    out = []
    for i, l in enumerate(model.layers):
        if l.is_full_attention:
            a = l.attention
            out.append(dict(layer=i, k=getattr(a, "k_caches", None), v=getattr(a, "v_caches", None),
                            block_size=getattr(model, "block_size", None) or getattr(model.args, "block_size", None)))
    return out


def gdn_descriptors(model):
    """Per GDN layer: (layer index, recurrent state, conv states). Used once the real GDN phases replace stand-ins."""
    out = []
    for i, l in enumerate(model.layers):
        if not l.is_full_attention:
            a = l.attention
            out.append(dict(layer=i, rec_state=getattr(a, "rec_state", None), conv_states=getattr(a, "conv_states", None)))
    return out


def per_replay_inputs(model, rows):
    """The fixed-address verify inputs the model stages each replay (addresses only go into runtime args)."""
    return dict(
        tokens=getattr(model, "_vfy_token_buf", None),
        kvpos=getattr(model, "_vfy_kvpos_buf", None),
        ctrl=getattr(model, "_vfy_ctrl", None),
        pt_rows=getattr(model, "_vfy_kvpt_buf", None),
        pt_users=getattr(model, "_vfy_kvpt_users", None),
        pt_groups=getattr(model, "_vfy_kvpt1_buf", None),
        state_blk_idx=getattr(model, "_vfy_state_idx", None),
        conv_sel=getattr(model, "_vfy_conv_sel", None),
        rows=rows,
    )


class Stage1Verify:
    def __init__(self, model):
        self.model = model
        self.layer_types = ["attention" if l.is_full_attention else "gdn" for l in model.layers]
        self.builder, self.reason = _engine_builder()
        self.weights_mk = None
        self.programs = {}  # n_users -> EngineProgram
        self.x_out = {}  # n_users -> persistent residual out
        self.ready = False

    def prepare(self, buckets, T):
        """Before capture: repack weights once, then build each bucket's engine program and its persistent x_out."""
        if self.builder is None:
            return {"ready": False, "reason": self.reason}
        from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk

        if self.weights_mk is None:
            self.weights_mk = weights_mk.repack(self.model)
        kv, gdn = kv_descriptors(self.model), gdn_descriptors(self.model)
        for b in buckets:
            prog = self.builder(self.model.mesh_device, self.layer_types, b, self.weights_mk, kv, gdn, None)
            alloc = getattr(prog, "alloc_x_out", None)
            if alloc is None:  # contract: the engine owns the residual layout (RES40 replicated), so it allocates x_out
                self.programs.clear()
                return {"ready": False, "reason": "EngineProgram has no alloc_x_out()"}
            self.programs[b], self.x_out[b] = prog, alloc()
        self.ready = True
        return {"ready": True, "buckets": sorted(self.programs)}

    def forward(self, x, n_users, T):
        """Inside the traced verify body: one generic_op replaces the layer loop. Returns the residual after the last
        layer, in the layout _spec_tail reads."""
        import ttnn

        prog = self.programs[n_users]
        if n_users not in self.x_out:
            raise RuntimeError("x_out must be allocated in prepare() (no allocation inside a trace capture)")
        io = prog.io(x, self.x_out[n_users], per_replay_inputs(self.model, n_users * T))
        ttnn.generic_op(io, prog.mesh_program)
        return self.x_out[n_users]
