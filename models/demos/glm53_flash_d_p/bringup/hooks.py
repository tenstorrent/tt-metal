# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for glm53_flash_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_state_pcc (KDA fixed-size state read-back), contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/glm_ref.py), the requested layers resident."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmReference

    return GlmReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF glm5_next code (vendored) on the dequantized checkpoint, text model only, routed experts read from
    disk per use. num_layers=None: the whole model in bf16 (intake sanity); else the first num_layers in fp32."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.hf_oracle import build_hf_model

    return build_hf_model(hf_path(spec), num_layers, torch.bfloat16 if num_layers is None else torch.float32)


# Device steps of the hybrid model, per block type: each passed its component gate on the device. Every other step
# runs on the CPU reference.
DEVICE_STEPS = {
    "kda_dense": {
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "attention",
        "attn_residual",
        "ffn_hc",
        "ffn_collapse",
        "ffn_norm",
        "mlp",
    },
    "dsa_moe": {
        "attn_hc",
        "attn_collapse",
        "attn_norm",
        "q_a",
        "indexer",
        "attention",
        "attn_residual",
        "ffn_hc",
        "ffn_collapse",
        "ffn_norm",
        "router",
        "experts",
        "shared_expert",
        "moe_add",
    },
    "kda_moe": set(),
}

_HC_STEPS = {"attn_hc": "attn", "ffn_hc": "ffn"}
_COLLAPSE_STEPS = {"attn_collapse", "ffn_collapse"}
_NORM_STEPS = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
_RESIDUAL_STEPS = {"attn_residual", "ffn_residual"}


def _loader_cfg(spec):
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig
    from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

    path = hf_path(spec)
    return WeightLoader(path), GlmConfig.from_json(os.path.join(path, "config.json"))


def _hc_host_fn(mesh, module, n):
    """fn(ctx, x_host [S * n, H]) -> host [S, (2 + n) * n] fp32 (harness boundary: upload bf16, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x):
        s = x.shape[0] // n
        xd = replicate(mesh, x.reshape(1, 1, s, n * x.shape[-1]).to(torch.bfloat16))
        yd = module(xd)
        y = replicated_to_host(yd).reshape(s, -1).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _collapse_host_fn(mesh, module, n):
    """fn(ctx, x_host [S * n, H], hc_host [S, K]) -> host [S, H] (harness boundary: x bf16, hc fp32, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x, hc):
        s = x.shape[0] // n
        xd = replicate(mesh, x.reshape(1, 1, s, n * x.shape[-1]).to(torch.bfloat16))
        hd = replicate(mesh, hc.reshape(1, 1, s, hc.shape[-1]).float(), dtype=ttnn.float32)
        yd = module(xd, hd)
        y = replicated_to_host(yd).reshape(s, -1)
        for t in (xd, hd, yd):
            ttnn.deallocate(t)
        return y

    return fn


def _residual_host_fn(mesh, module, n):
    """fn(ctx, x_host [S * n, H], hc_host [S, K], y_host [S, H]) -> host [S * n, H] bf16 (harness boundary: x, y bf16,
    hc fp32, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x, hc, y):
        s = x.shape[0] // n
        xd = replicate(mesh, x.reshape(1, 1, s, n * x.shape[-1]).to(torch.bfloat16))
        hd = replicate(mesh, hc.reshape(1, 1, s, hc.shape[-1]).float(), dtype=ttnn.float32)
        yd = replicate(mesh, y.reshape(1, 1, s, y.shape[-1]).to(torch.bfloat16))
        od = module(xd, hd, yd)
        out = replicated_to_host(od).reshape(s * n, -1)
        for t in (xd, hd, yd, od):
            ttnn.deallocate(t)
        return out

    return fn


def _norm_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] bf16 (harness boundary: upload bf16, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x):
        xd = replicate(mesh, x.reshape(1, 1, *x.shape[-2:]).to(torch.bfloat16))
        yd = module(xd)
        y = replicated_to_host(yd).reshape(x.shape[-2], -1)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _add_host_fn(mesh, module):
    """fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H] bf16 (harness boundary: upload bf16, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, a, b):
        s = a.shape[-2]
        ad = replicate(mesh, a.reshape(1, 1, s, a.shape[-1]).to(torch.bfloat16))
        bd = replicate(mesh, b.reshape(1, 1, s, b.shape[-1]).to(torch.bfloat16))
        yd = module(ad, bd)
        y = replicated_to_host(yd).reshape(s, -1)
        for t in (ad, bd, yd):
            ttnn.deallocate(t)
        return y

    return fn


def _router_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> dense routing host [S, E] bf16 around TtRouter (harness boundary: bf16 upload, chip-0
    read-back)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x):
        s = x.shape[-2]
        xd = replicate(mesh, x.reshape(1, 1, s, x.shape[-1]).to(torch.bfloat16))
        dense, idx, wts = module(xd)
        y = replicated_to_host(dense).reshape(s, -1)
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return y

    return fn


def _experts_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H] bf16 around TtExperts
    (harness boundary: bf16 upload, chip-0 read-back)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x, r):
        s = x.shape[-2]
        xd = replicate(mesh, x.reshape(1, 1, s, x.shape[-1]).to(torch.bfloat16))
        rd = replicate(mesh, r.reshape(1, 1, s, r.shape[-1]).to(torch.bfloat16))
        yd = module(xd, dense=rd)
        y = replicated_to_host(yd).reshape(s, -1)
        for t in (xd, rd, yd):
            ttnn.deallocate(t)
        return y

    return fn


class _KdaHostFn:
    """fn(ctx, x_host [S, H]) -> host [S, H] bf16 around TtKdaAttention (harness boundary: bf16 upload, chip-0
    read-back). A component ctx that carries ``state_prefix`` loads it first and gets ``state_out`` back; otherwise
    the module's own carried state continues from the previous chunk."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def __call__(self, ctx, x):
        import ttnn
        from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

        prefix = ctx.extra.get("state_prefix")
        if prefix is not None and ctx.start > 0:
            self.module.load_state(prefix)
        s = x.shape[-2]
        xd = replicate(self.mesh, x.reshape(1, 1, s, x.shape[-1]).to(torch.bfloat16))
        yd = self.module(xd, ctx.start)
        y = replicated_to_host(yd).reshape(s, -1)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        if prefix is not None:
            ctx.extra["state_out"] = self.module.state_torch()
        return y

    def load_state(self, tensors, length=None):
        self.module.load_state(tensors)

    def state_torch(self):
        return self.module.state_torch()


class _IndexerHostFn:
    """fn(ctx, attn_norm [S, H], q_resid [S, 1536]) -> host topk int32 [S, 2051] (-1 = none) around TtIndexer
    (harness boundary: bf16 upload, read-back of the four per-chip row blocks, concatenated on the host). A
    component ctx that carries ``state_prefix`` loads its pooled keys first and gets ``state_out`` back."""

    def __init__(self, mesh, module, width):
        self.mesh, self.module, self.width = mesh, module, width

    def __call__(self, ctx, x, q_resid):
        import ttnn
        from models.demos.glm53_flash_d_p.tt.common import replicate

        prefix = ctx.extra.get("state_prefix")
        if prefix is not None and ctx.start > 0:
            self.module.load_state(prefix, ctx.extra.get("prefix_len", ctx.start))
        s = x.shape[-2]
        xd = replicate(self.mesh, x.reshape(1, 1, s, x.shape[-1]).to(torch.bfloat16))
        qd = replicate(self.mesh, q_resid.reshape(1, 1, s, q_resid.shape[-1]).to(torch.bfloat16))
        td = self.module(xd, qd, ctx.start)
        # chip d = 2 r + c holds rows d S/4 ..; get_device_tensors is in that (row-major) order
        y = torch.cat(
            [ttnn.to_torch(t).reshape(s // len(ttnn.get_device_tensors(td)), -1) for t in ttnn.get_device_tensors(td)]
        )
        y = y.view(torch.int32) if y.dtype == torch.uint32 else y.to(torch.int32)
        y = y[:, : self.width].contiguous()
        for t in (xd, qd, td):
            ttnn.deallocate(t)
        if prefix is not None:
            ctx.extra["state_out"] = self.module.state_torch()
        return y

    def load_state(self, tensors, length=None):
        self.module.load_state(tensors, length)

    def state_torch(self):
        return self.module.state_torch()


class _MlaHostFn:
    """fn(ctx, attn_norm [S, H], q_resid [S, 1536], topk int32 [S, 2051]) -> host [S, H] bf16 around TtMLA (harness
    boundary: bf16 upload, topk compacted to the sparse_sdpa row format and split by chip, chip-0 read-back). A
    component ctx that carries ``state_prefix`` loads its latent rows first and gets ``state_out`` back."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def __call__(self, ctx, x, q_resid, topk):
        import ttnn
        from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host
        from models.demos.glm53_flash_d_p.tt.mla_attention import idx_to_device

        prefix = ctx.extra.get("state_prefix")
        if prefix is not None and ctx.start > 0:
            self.module.load_state(prefix, ctx.extra.get("prefix_len", ctx.start))
        s = x.shape[-2]
        xd = replicate(self.mesh, x.reshape(1, 1, s, x.shape[-1]).to(torch.bfloat16))
        qd = replicate(self.mesh, q_resid.reshape(1, 1, s, q_resid.shape[-1]).to(torch.bfloat16))
        idx = idx_to_device(self.mesh, topk.reshape(s, -1))
        yd = self.module(xd, qd, idx, ctx.start)
        y = replicated_to_host(yd).reshape(s, -1)
        for t in (xd, qd, idx, yd):
            ttnn.deallocate(t)
        if prefix is not None:
            ctx.extra["state_out"] = self.module.state_torch()
        return y

    def load_state(self, tensors, length=None):
        self.module.load_state(tensors, length)

    def state_torch(self):
        return self.module.state_torch()


def _max_seq(spec):
    seqs = [r["seq"] for r in spec.get("ladder", [])] + [spec.get("target", {}).get("seq", 0)]
    return max(seqs)


def _chunks(spec):
    return sorted(({r["chunk"] for r in spec.get("ladder", [])} | {spec.get("target", {}).get("chunk", 0)}) - {0})


def _device_step(mesh, spec, layer, step, loader, cfg):
    if step == "attention" and cfg.is_kda(layer):
        from models.demos.glm53_flash_d_p.tt.kda_attention import build_kda_attention

        return _KdaHostFn(mesh, build_kda_attention(mesh, loader, cfg, layer, _max_seq(spec)))
    if step in _HC_STEPS:
        from models.demos.glm53_flash_d_p.tt.mhc import build_hc

        return _hc_host_fn(mesh, build_hc(mesh, loader, cfg, layer, _HC_STEPS[step]), cfg.hc_mult)
    if step in _COLLAPSE_STEPS:
        from models.demos.glm53_flash_d_p.tt.collapse import build_collapse

        return _collapse_host_fn(mesh, build_collapse(cfg), cfg.hc_mult)
    if step in _RESIDUAL_STEPS:
        from models.demos.glm53_flash_d_p.tt.residual import build_residual

        return _residual_host_fn(mesh, build_residual(cfg, mesh), cfg.hc_mult)
    if step in _NORM_STEPS:
        from models.demos.glm53_flash_d_p.tt.rms_norm import build_norm

        return _norm_host_fn(mesh, build_norm(mesh, loader, cfg, layer, _NORM_STEPS[step]))
    if step == "q_a":
        from models.demos.glm53_flash_d_p.tt.q_a import build_q_a

        return _norm_host_fn(mesh, build_q_a(mesh, loader, cfg, layer))
    if step == "indexer":
        from models.demos.glm53_flash_d_p.tt.indexer import build_indexer

        module = build_indexer(mesh, loader, cfg, layer, _max_seq(spec), _chunks(spec))
        return _IndexerHostFn(mesh, module, cfg.index_topk + cfg.index_kpool - 1)
    if step == "attention":
        from models.demos.glm53_flash_d_p.tt.mla_attention import build_mla

        return _MlaHostFn(mesh, build_mla(mesh, loader, cfg, layer, _max_seq(spec)))
    if step == "mlp" and not cfg.is_moe(layer):
        from models.demos.glm53_flash_d_p.tt.mlp import build_mlp

        return _norm_host_fn(mesh, build_mlp(mesh, loader, cfg, layer))
    if step == "shared_expert":
        from models.demos.glm53_flash_d_p.tt.mlp import build_mlp

        return _norm_host_fn(mesh, build_mlp(mesh, loader, cfg, layer, name="mlp.shared_experts"))
    if step == "router":
        from models.demos.glm53_flash_d_p.tt.router import build_router

        return _router_host_fn(mesh, build_router(mesh, loader, cfg, layer, max(_chunks(spec))))
    if step == "experts":
        from models.demos.glm53_flash_d_p.tt.experts import build_experts

        return _experts_host_fn(mesh, build_experts(mesh, loader, cfg, layer, max(_chunks(spec))))
    if step == "moe_add":
        from models.demos.glm53_flash_d_p.tt.moe_add import build_moe_add

        return _add_host_fn(mesh, build_moe_add(cfg))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_component(mesh, spec, layer, step):
    loader, cfg = _loader_cfg(spec)
    return _device_step(mesh, spec, layer, step, loader, cfg)


class _RefState:
    """CPU reference state; a layer whose stateful step runs on the device keeps its state there (``dev``)."""

    def __init__(self, ref, max_seq, dev=None):
        self.ref, self.s, self.dev = ref, ref.new_state(max_seq), dev or {}

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)
        for d in self.dev.get(layer, ()):
            d.load_state(tensors, length)

    def to_torch(self, layer, length):
        """The CPU state with the device-held tensors (a KDA layer's whole state, a DSA layer's pooled keys and
        latent cache) on top."""
        out = self.ref.state_tensors(self.s, layer, length)
        for d in self.dev.get(layer, ()):
            out.update(d.state_torch())
        if "index_key" in out:
            out["index_key"] = out["index_key"][: length // self.ref.cfg.index_kpool]
        if "kv_latent" in out:
            out["kv_latent"] = out["kv_latent"][:length]
        return out


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Selected with BRINGUP_HYBRID=1 (debugging); the default device model is GlmDeviceModel."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        t0 = time.time()
        self.mesh, self.spec = mesh, spec
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader, cfg = _loader_cfg(spec)
        self.overrides = {
            i: {s: _device_step(mesh, spec, i, s, loader, cfg) for s in DEVICE_STEPS.get(spec.block_type_of(i), ())}
            for i in self.ref.layer_ids
        }
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        dev = {i: [f for f in o.values() if hasattr(f, "state_torch")] for i, o in self.overrides.items()}
        return _RefState(self.ref, max_seq, {i: fs for i, fs in dev.items() if fs})

    def embed(self, tokens):
        import torch.nn.functional as F

        e = F.embedding(tokens.long(), self.ref.embed)
        return e.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).reshape(-1, e.shape[-1])

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
        from models.demos.glm53_flash_d_p.reference.glm_ref import rms_norm

        n = self.cfg.hc_mult
        return rms_norm(h.view(-1, n, h.shape[-1]).mean(dim=1), self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


class _DeviceState:
    """Per-layer state held by the device blocks (KDA carries, indexer pooled keys, MLA latent cache)."""

    def __init__(self, blocks):
        self.blocks = blocks

    def load_prefix(self, layer, tensors, length):
        self.blocks[layer].load_state(tensors, length)

    def to_torch(self, layer, length):
        return self.blocks[layer].state_torch(length)


class GlmDeviceModel:
    """Ladder / profile adapter over tt/model.py:TtGlmModel.

    The residual is a replicated [1, 1, S, 4 H] bf16 device tensor (4 mHC streams packed on the last dim) from the
    embedding to the final norm. Each layer is TtGlmBlock.__call__: run_block over the reference block graph with the
    validated device modules (one profiler section per step). Only the LM head runs on the host (ladder logits on
    sampled rows, when the stack ends at the model's last layer)."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.common.bringup.reference.golden import hf_path
        from models.demos.glm53_flash_d_p.tt.model import TtGlmModel

        t0 = time.time()
        self.mesh, self.spec = mesh, spec
        self.path = hf_path(spec)
        self.model = TtGlmModel(mesh, self.path, max_seq=_max_seq(spec), chunks=_chunks(spec), layers=list(layers))
        self.cfg = self.model.cfg
        self.n = self.cfg.hc_mult
        self.blocks = {b.i: b for b in self.model.blocks}
        self._lm_head = None
        if lm_head:
            from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

            self._lm_head = WeightLoader(self.path).get("lm_head.weight").float()  # untied
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _DeviceState(self.blocks)

    def embed(self, tokens):
        import ttnn

        ids = ttnn.from_torch(
            tokens.reshape(1, 1, -1).to(torch.int64).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        h = self.model.embed(ids)
        ttnn.deallocate(ids)
        return h

    def from_host(self, h):
        """Reference residual [S * 4, H] -> device [1, 1, S, 4 H] (harness boundary)."""
        from models.demos.glm53_flash_d_p.tt.common import replicate

        s = h.shape[0] // self.n
        return replicate(self.mesh, h.reshape(1, 1, s, self.n * h.shape[-1]).to(torch.bfloat16))

    def to_host(self, h):
        """Device [1, 1, S, 4 H] -> reference [S * 4, H]; [1, 1, S, H] (the final norm) -> [S, H]."""
        from models.demos.glm53_flash_d_p.tt.common import replicated_to_host

        t = replicated_to_host(h).float()
        s, w = t.shape[-2], t.shape[-1]
        hidden = self.cfg.hidden_size
        return t.reshape(s * (w // hidden), hidden)

    def layer(self, i, h, start, state):
        return self.blocks[i](h, start)

    def final_norm(self, h):
        return self.model.final_norm(h)

    def logits(self, hidden, rows):
        return torch.nn.functional.linear(self.to_host(hidden)[rows], self._lm_head)

    def free(self, h):
        import ttnn

        if isinstance(h, ttnn.Tensor) and h.is_allocated():
            ttnn.deallocate(h)

    def sync(self):
        import ttnn

        ttnn.synchronize_device(self.mesh)

    def perf_settings(self):
        """Recorded in the profile: the mHC residual mix path (GLM_RESIDUAL_MIX=matmul | addcmul)."""
        from models.demos.glm53_flash_d_p.tt.residual import residual_mix_mode

        return {"residual_mix": residual_mix_mode()}


def device_model(mesh, spec, layers, lm_head=True):
    """All-device model (default); BRINGUP_HYBRID=1 selects the hybrid harness (CPU reference + DEVICE_STEPS on the
    device, host in / host out per step) for debugging."""
    import os

    if os.environ.get("BRINGUP_HYBRID") == "1":
        return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
    return GlmDeviceModel(mesh, spec, layers, lm_head=lm_head)


def contract_state_pcc(spec, runtime, kv, slot, length, golden):
    """The slot's KDA carries (spec state.fixed) after a request of ``length`` tokens vs the golden snapshot."""
    from models.demos.glm53_flash_d_p.tt.runners.adapter import contract_state_pcc as _impl

    return _impl(spec, runtime, kv, slot, length, golden)
