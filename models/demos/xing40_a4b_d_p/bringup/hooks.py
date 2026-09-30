# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for xing40_a4b_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def reference(spec, layers=None, dtype=None):
    """The standalone chunked CPU reference (reference/xing_ref.py): mHC 4-stream residual, dense causal MLA with a
    latent cache, DeepSeek-style MoE. All requested layers stay resident (40 layers: 122 GB in fp32). The HF oracle
    is the stock loader (trust_remote_code); its [B, S, 4, H] layer output flattens to the reference's [S * 4, H]
    block output, so check_hf needs no hf_model / hf_layers hook."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.xing_ref import XingReference

    return XingReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


# Device steps of the hybrid harness, per block type: every step passed its component gate on the device (and its
# swap gate, once run). Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "dense": {"attn_hc", "attn_collapse", "attn_norm", "q_a", "attention", "attn_residual", "ffn_norm", "mlp"},
    "moe": {"router", "experts", "shared_expert", "moe_add"},
}

# mHC coefficient steps (tt/mhc.py:TtHcWeights) -> checkpoint prefix under model.layers.<i>.
_HC_STEPS = {"attn_hc", "ffn_hc"}
# mHC collapse steps (tt/collapse.py:TtHcCollapse): (streams, hc) -> [S, H], no weights.
_COLLAPSE_STEPS = {"attn_collapse", "ffn_collapse"}
# Distributed RMSNorm steps (tt/norm.py:TtDistributedRmsNorm): column-split [S, H] in -> column-split [S, H] out.
_NORM_STEPS = {"attn_norm"}
# ffn_norm: same norm, then all_gather over axis 1 -> row-split [S, H], replicated over axis 1.
_GATHERED_NORM_STEPS = {"ffn_norm"}
# q_a stem (tt/q_a.py:TtQa): column-split [S, H] in -> K-split q_a_proj -> all_reduce axis 1 -> q_a_layernorm ->
# row-split q_resid [S, 768], replicated over axis 1.
_QA_STEPS = {"q_a"}
# Dense causal MLA (tt/attention.py:TtMlaAttention), stateful: owns the layer's block-cyclic device latent cache.
_ATTENTION_STEPS = {"attention"}
# mHC residual mix (tt/residual.py:TtHcResidual): (streams, hc, y column-split) -> streams, no weights, no CCL.
_RESIDUAL_STEPS = {"attn_residual", "ffn_residual"}
# Dense SwiGLU (tt/mlp.py:TtDenseMLP): row-split ffn_norm [S, H] (replicated over axis 1) in -> gate / up
# column-parallel, down row-parallel -> reduce_scatter axis 1 -> column-split mlp_out [S, H] fp32.
_MLP_STEPS = {"mlp"}
# MoE shared expert: the same TtDenseMLP (SwiGLU 1024, 512 per column) on the mlp.shared_experts.* weights, same
# boundary as mlp (row-split ffn_norm in, column-split shared_out fp32 out).
_SHARED_EXPERT_STEPS = {"shared_expert"}
# MoE router (tt/router.py:TtRouter): row-split ffn_norm [S, H] (replicated over axis 1) in -> replicated fp32 gate,
# sigmoid, bias, top-4, renorm x 2.0 -> row-split dense routing [S, 64] fp32 (+ idx / weights for dispatch). No CCL.
_ROUTER_STEPS = {"router"}
# Routed experts (tt/experts.py:TtExperts): row-split ffn_norm [S, H] bf16 + the router's (idx, wts) -> dispatch /
# combine over axis 0 (dispatch groups = columns) -> reduce_scatter axis 1 -> column-split experts_out [S, H] fp32.
_EXPERTS_STEPS = {"experts"}
# moe_add (tt/moe_add.py:TtMoeAdd): experts_out + shared_out, both column-split [S, H] fp32 -> column-split mlp_out
# fp32. Local, no CCL.
_MOE_ADD_STEPS = {"moe_add"}


def _max_rows(spec):
    """Largest per-chip row count of any chunk the spec runs (ladder rungs and the target), over SP = mesh rows."""
    chunks = [r["chunk"] for r in (spec.get("ladder") or []) if "chunk" in r]
    tgt = spec.get("target") or {}
    if "chunk" in tgt:
        chunks.append(tgt["chunk"])
    return max(chunks) // spec.mesh[0]


def _max_chunk(spec):
    """Largest chunk (all mesh rows together) of any ladder rung or the target."""
    return _max_rows(spec) * spec.mesh[0]


def _loader(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.weights import WeightLoader

    return WeightLoader(hf_path(spec))


def _cfg(spec):
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.xing_ref import XingConfig

    return XingConfig.from_json(os.path.join(hf_path(spec), "config.json"))


def _hc_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S * 4, H]) -> coefficients host [S, 24] fp32 (harness boundary: streams in, out back)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import row_split_to_host, streams_to_device

    def fn(ctx, x):
        xd = streams_to_device(mesh, x, hidden)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _collapse_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S * 4, H], hc_host [S, 24]) -> host [S, H] fp32 (harness boundary)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_host, row_split_to_device, streams_to_device

    def fn(ctx, x, hc):
        xd = streams_to_device(mesh, x, hidden)
        hd = row_split_to_device(mesh, hc)
        od = module(xd, hd)
        out = col_split_to_host(mesh, od).float()
        for t in (xd, hd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _norm_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 (harness boundary: column-split in, column-split out)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, col_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x)
        od = module(xd)
        out = col_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _gathered_norm_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 (harness boundary: column-split in, the row-split output
    replicated over axis 1 read back from column 0)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _qa_host_fn(mesh, module):
    """fn(ctx, attn_norm_host [S, H]) -> q_resid host [S, 768] fp32 (harness boundary: column-split bf16 in, the
    row-split output replicated over axis 1 read back from column 0)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _residual_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S * 4, H], hc_host [S, 24], y_host [S, H]) -> streams host [S * 4, H] fp32 (harness
    boundary: streams / hc / column-split y in, streams back)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, row_split_to_device, streams_to_device

    def fn(ctx, x, hc, y):
        from models.demos.xing40_a4b_d_p.tt.layout import streams_to_host

        xd = streams_to_device(mesh, x, hidden)
        hd = row_split_to_device(mesh, hc)
        yd = col_split_to_device(mesh, y)
        od = module(xd, hd, yd)
        out = streams_to_host(mesh, od, hidden).float()
        for t in (xd, hd, yd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _mlp_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H]) -> mlp_out host [S, H] fp32 (harness boundary: row-split bf16 in, replicated
    over axis 1; column-split out)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_host, row_split_to_device

    def fn(ctx, x):
        xd = row_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        od = module(xd)
        out = col_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _moe_add_host_fn(mesh, module):
    """fn(ctx, experts_out_host [S, H], shared_out_host [S, H]) -> mlp_out host [S, H] fp32 (harness boundary: both
    column-split fp32 in, as TtExperts / TtDenseMLP return them; column-split out)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, col_split_to_host

    def fn(ctx, a, b):
        ad = col_split_to_device(mesh, a, dtype=ttnn.float32)
        bd = col_split_to_device(mesh, b, dtype=ttnn.float32)
        od = module(ad, bd)
        out = col_split_to_host(mesh, od).float()
        for t in (ad, bd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _router_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H]) -> dense routing host [S, E] fp32 (harness boundary: row-split bf16 in,
    replicated over axis 1, as ffn_norm's device output; the row-split dense matrix read back from column 0)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import row_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = row_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        dense, idx, wts = module(xd)
        out = row_split_to_host(mesh, dense).float()
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _experts_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H] fp32 (harness boundary:
    the dense routing is turned back into the router's (idx uint16, wts fp32) [S, K] here, row-split over axis 0 and
    replicated over axis 1 like x and like TtRouter's outputs; the column-split output is read back)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_host, row_split_to_device

    def fn(ctx, x, routing):
        s = x.shape[0]
        tw, ti = torch.topk(routing.float(), k=module.K, dim=-1, sorted=True)
        xd = row_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        idd = ttnn.from_torch(
            ti.to(torch.int32).reshape(1, 1, s, module.K),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
        )
        wd = row_split_to_device(mesh, tw, dtype=ttnn.float32)
        yd = module(xd, idd, wd)
        y = col_split_to_host(mesh, yd).float()
        for t in (xd, idd, wd, yd):
            ttnn.deallocate(t)
        return y

    fn.module = module
    return fn


class _AttentionHostFn:
    """fn(ctx, attn_norm_host [S, H], q_resid_host [S, 768]) -> attn_out host [S, H] fp32.

    Harness boundary around TtMlaAttention, which keeps the layer's MLA latent cache on the device. With a device
    ctx (component / swap tests: ``state_prefix``, ``prefix_len``, ``max_seq`` in ctx.extra) every call reloads the
    golden kv_latent prefix. In the hybrid model the cache persists across chunks; the hybrid state calls ``reset``
    / ``load_prefix`` / ``read_state``."""

    stateful = True
    state_key = "kv_latent"

    def __init__(self, mesh, module):
        self.mesh, self.mod = mesh, module
        self.reset()

    def reset(self):
        self._pending, self._fresh = None, True

    def load_prefix(self, kv_latent):
        self._pending, self._fresh = kv_latent, True

    def read_state(self, length):
        return self.mod.read_state(length)

    def __call__(self, ctx, x, q_resid):
        import ttnn
        from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, col_split_to_host, row_split_to_device

        if "state_prefix" in ctx.extra:
            self.mod.setup(ctx.length, ctx.extra["max_seq"])
            self.mod.load_state(ctx.extra["state_prefix"]["kv_latent"][: ctx.extra["prefix_len"]])
        else:
            self.mod.setup(ctx.length, ctx.state.max_seq)
            if self._fresh:
                self.mod.load_state(self._pending)
                self._fresh = False
        xd = col_split_to_device(self.mesh, x, dtype=ttnn.bfloat16)
        qd = row_split_to_device(self.mesh, q_resid, dtype=ttnn.bfloat16)
        od = self.mod(xd, qd, ctx.start)
        out = col_split_to_host(self.mesh, od).float()
        for t in (xd, qd, od):
            ttnn.deallocate(t)
        return out


def _device_step_fn(mesh, spec, layer, step, loader, cfg):
    if step in _ATTENTION_STEPS:
        from models.demos.xing40_a4b_d_p.tt.attention import build_attention

        return _AttentionHostFn(mesh, build_attention(mesh, loader, cfg, layer))
    if step in _RESIDUAL_STEPS:
        from models.demos.xing40_a4b_d_p.tt.residual import build_residual

        return _residual_host_fn(mesh, build_residual(cfg, mesh), cfg.hidden_size)
    if step in _COLLAPSE_STEPS:
        from models.demos.xing40_a4b_d_p.tt.collapse import build_collapse

        return _collapse_host_fn(mesh, build_collapse(cfg), cfg.hidden_size)
    if step in _HC_STEPS:
        from models.demos.xing40_a4b_d_p.tt.mhc import build_hc

        return _hc_host_fn(mesh, build_hc(mesh, loader, cfg, layer, step), cfg.hidden_size)
    if step in _NORM_STEPS:
        from models.demos.xing40_a4b_d_p.tt.norm import build_norm

        return _norm_host_fn(mesh, build_norm(mesh, loader, cfg, layer, step))
    if step in _GATHERED_NORM_STEPS:
        from models.demos.xing40_a4b_d_p.tt.norm import build_norm

        return _gathered_norm_host_fn(mesh, build_norm(mesh, loader, cfg, layer, step))
    if step in _MLP_STEPS:
        from models.demos.xing40_a4b_d_p.tt.mlp import build_mlp

        return _mlp_host_fn(mesh, build_mlp(mesh, loader, cfg, layer))
    if step in _SHARED_EXPERT_STEPS:
        from models.demos.xing40_a4b_d_p.tt.mlp import build_mlp

        return _mlp_host_fn(mesh, build_mlp(mesh, loader, cfg, layer, prefix="mlp.shared_experts."))
    if step in _ROUTER_STEPS:
        from models.demos.xing40_a4b_d_p.tt.router import build_router

        return _router_host_fn(mesh, build_router(mesh, loader, cfg, layer, _max_rows(spec)))
    if step in _EXPERTS_STEPS:
        from models.demos.xing40_a4b_d_p.tt.experts import build_experts

        return _experts_host_fn(mesh, build_experts(mesh, loader, cfg, layer, _max_chunk(spec)))
    if step in _MOE_ADD_STEPS:
        from models.demos.xing40_a4b_d_p.tt.moe_add import build_moe_add

        return _moe_add_host_fn(mesh, build_moe_add(cfg))
    if step in _QA_STEPS:
        from models.demos.xing40_a4b_d_p.tt.q_a import build_q_a

        return _qa_host_fn(mesh, build_q_a(mesh, loader, cfg, layer))
    return None


def device_component(mesh, spec, layer, step):
    fn = _device_step_fn(mesh, spec, layer, step, _loader(spec), _cfg(spec))
    if fn is None:
        raise NotImplementedError(f"implement step: no device module for {step} yet")
    return fn


class _HybridState:
    """The CPU reference state, with the caches of the device's stateful steps (``_AttentionHostFn``: kv_latent) on
    the device."""

    def __init__(self, ref, max_seq, device_state=None):
        self.ref, self.s = ref, ref.new_state(max_seq)
        self.dev = device_state or {}  # layer -> [stateful host fns], each owning ``state_key``
        for fns in self.dev.values():
            for fn in fns:
                fn.reset()

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)
        for fn in self.dev.get(layer, ()):
            fn.load_prefix(tensors[fn.state_key][:length])

    def to_torch(self, layer, length):
        d = self.ref.state_tensors(self.s, layer, length)
        for fn in self.dev.get(layer, ()):
            d[fn.state_key] = fn.read_state(length)
        return d


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Hidden states (the 4 mHC streams, [S * 4, H] fp32 token-major) stay on the host. Selected with BRINGUP_HYBRID=1
    (debugging); the default device model is XingDeviceModel."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = _loader(spec)
        self.overrides = {}
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            self.overrides[i] = {s: _device_step_fn(mesh, spec, i, s, loader, self.cfg) for s in steps}
            missing = [s for s, f in self.overrides[i].items() if f is None]
            assert not missing, f"layer {i}: no device module for {missing}"
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        dev = {i: [f for f in o.values() if getattr(f, "stateful", False)] for i, o in self.overrides.items()}
        return _HybridState(self.ref, max_seq, dev)

    def embed(self, tokens):
        import torch.nn.functional as F

        e = F.embedding(tokens.long().reshape(-1), self.ref.embed)
        return e.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).reshape(-1, e.shape[-1]).contiguous()

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0] // self.cfg.hc_mult, state.s)
        return run_block(
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        from models.demos.xing40_a4b_d_p.reference.xing_ref import rms_norm

        mean = h.view(-1, self.cfg.hc_mult, h.shape[-1]).mean(dim=1)
        return rms_norm(mean, self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def _chunks_for(spec, max_seq):
    """Chunk lengths of the ladder rungs / target that run a sequence of ``max_seq`` (first = the default geometry)."""
    runs = list(spec.get("ladder") or []) + [spec.get("target") or {}]
    out = [r["chunk"] for r in runs if r.get("seq") == max_seq and "chunk" in r]
    return list(dict.fromkeys(out))


class _DeviceState:
    """Per-layer state held by the device blocks (the MLA latent cache in each TtMlaAttention). Created outside the
    forward: sets up every block's geometry for the chunk this sequence runs at and zeroes the caches."""

    def __init__(self, model, max_seq):
        self.blocks = model.blocks  # layer -> TtXingBlock
        self.max_seq = max_seq
        chunks = _chunks_for(model.spec, max_seq)
        assert chunks, f"no ladder rung / target runs seq {max_seq}"
        for c in reversed(chunks):  # the first listed ends up current
            model.model.setup(c, max_seq)
        for b in self.blocks.values():
            b.load_state(None)

    def load_prefix(self, layer, tensors, length):
        self.blocks[layer].load_state(tensors["kv_latent"][:length])

    def to_torch(self, layer, length):
        return self.blocks[layer].state_torch(length)


class XingDeviceModel:
    """Ladder / profile adapter over tt/model.py:TtXingModel.

    The residual (4 mHC streams, [1, 1, S/4, 4 x 1792] fp32 per chip: rows over axis 0, hidden columns over axis 1)
    stays on the device from the embedding to the final norm. Each layer is TtXingBlock.__call__: run_block over the
    reference block graph with the validated device modules (one profiler section per step). Host work per chunk:
    the token ids in (embed) and the harness read-backs; the LM head runs on the host on the ladder's sampled rows."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.common.bringup.reference.golden import hf_path
        from models.demos.xing40_a4b_d_p.tt.model import TtXingModel

        t0 = time.time()
        self.mesh, self.spec = mesh, spec
        self.path = hf_path(spec)
        self.model = TtXingModel(mesh, self.path, _max_rows(spec), _max_chunk(spec), layers=list(layers))
        self.cfg = self.model.cfg
        self.blocks = {b.i: b for b in self.model.blocks}
        self._lm_head = _loader(spec).get("lm_head.weight").float() if lm_head else None
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _DeviceState(self, max_seq)

    def embed(self, tokens):
        import ttnn

        ids = self.model.embed.ids_to_device(tokens)
        h = self.model.embed(ids)
        ttnn.deallocate(ids)
        return h

    def from_host(self, h):
        """Reference streams [S * 4, H] -> device [1, 1, S/4, 4 x 1792] fp32 (harness boundary)."""
        from models.demos.xing40_a4b_d_p.tt.layout import streams_to_device

        return streams_to_device(self.mesh, h, self.cfg.hidden_size)

    def to_host(self, h):
        """Streams -> [S * 4, H]; the final norm's [1, 1, S/4, 1792] column split -> [S, H]."""
        from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_host, streams_to_host

        if h.shape[-1] == self.cfg.hc_mult * self.cfg.hidden_size // self.mesh.shape[1]:
            return streams_to_host(self.mesh, h, self.cfg.hidden_size)
        return col_split_to_host(self.mesh, h)

    def layer(self, i, h, start, state):
        return self.blocks[i](h, start)

    def final_norm(self, h):
        return self.model.final_norm(h)

    def logits(self, hidden, rows):
        return torch.nn.functional.linear(self.to_host(hidden).float()[rows], self._lm_head)

    def free(self, h):
        import ttnn

        if isinstance(h, ttnn.Tensor) and h.is_allocated():
            ttnn.deallocate(h)

    def sync(self):
        import ttnn

        ttnn.synchronize_device(self.mesh)

    def perf_settings(self):
        """Recorded in the profile: the mHC residual mix (XING_RESIDUAL_MIX), the ring_mla implementation
        (XING_MLA_SDPA), its fidelity / chunks (XING_MLA_SDPA_FIDELITY, XING_MLA_Q_CHUNK) and the routed-experts
        path (XING_EXPERTS_MODE)."""
        import os

        from models.demos.xing40_a4b_d_p.tt.attention import K_CHUNK, sdpa_fidelity, sdpa_impl, sdpa_q_chunk
        from models.demos.xing40_a4b_d_p.tt.residual import residual_mix_mode

        return {
            "residual_mix": residual_mix_mode(),
            "mla_sdpa": sdpa_impl(),
            "mla_sdpa_fidelity": sdpa_fidelity(),
            "mla_sdpa_chunks": f"q{sdpa_q_chunk()}/k{K_CHUNK}",
            "experts_mode": os.environ.get("XING_EXPERTS_MODE", "unified"),
        }


def device_model(mesh, spec, layers, lm_head=True):
    """All-device model (default); BRINGUP_HYBRID=1 selects the hybrid harness (CPU reference + DEVICE_STEPS on the
    device, host in / host out per step) for debugging."""
    import os

    if os.environ.get("BRINGUP_HYBRID") == "1":
        return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
    return XingDeviceModel(mesh, spec, layers, lm_head=lm_head)
