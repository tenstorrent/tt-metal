# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gemma4_a4b_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""


import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import Gemma4Reference

    return Gemma4Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_layers(model):
    """Decoder layers of the HF oracle (Gemma4ForConditionalGeneration: model.model.language_model.layers)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


# Device steps swapped in so far, per block type. Every other step runs on the CPU reference.
DEVICE_STEPS = {
    "sliding": {
        "attn_norm",
        "attention",
        "post_attn_norm",
        "attn_residual",
        "ffn_norm",
        "mlp",
        "post_mlp_norm",
        "router",
        "moe_norm",
        "experts",
    },
    "global": set(),
}

# Residual steps: h_mid = in + attn_post_norm (replicated, no collective).
_RESIDUAL_STEPS = {"attn_residual"}

# Norm steps -> checkpoint weight name (under model.language_model.layers.<i>.).
_NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "post_attn_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight",
    "post_mlp_norm": "post_feedforward_layernorm_1.weight",
    "moe_norm": "pre_feedforward_layernorm_2.weight",
}


def _norm_module(mesh, spec, layer, step, loader=None):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, WeightLoader
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import TtRMSNorm

    loader = loader or WeightLoader(hf_path(spec))
    w = loader.get(f"{PREFIX}layers.{layer}.{_NORM_WEIGHTS[step]}")
    return TtRMSNorm(mesh, w, eps=1e-6)


def _host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H]."""
    import ttnn
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd)
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def _residual_host_fn(mesh):
    """fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H] via TtResidualAdd."""
    import ttnn
    from models.demos.gemma4_a4b_d_p.tt.residual import TtResidualAdd
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    module = TtResidualAdd(mesh)

    def fn(ctx, a, b):
        ad = to_device_replicated(mesh, a)
        bd = to_device_replicated(mesh, b)
        yd = module(ad, bd)
        y = replicated_to_host(yd)
        for t in (ad, bd, yd):
            ttnn.deallocate(t)
        return y.to(a.dtype)

    return fn


def _mlp_module(mesh, spec, layer, loader=None):
    """TtDenseMLP (TP=4, intermediate padded 528 -> 544 per chip) for one layer's dense mlp."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, WeightLoader
    from models.demos.gemma4_a4b_d_p.tt.mlp import TtDenseMLP

    loader = loader or WeightLoader(hf_path(spec))
    p = f"{PREFIX}layers.{layer}.mlp."
    return TtDenseMLP(mesh, *(loader.get(p + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj")))


def _router_module(mesh, spec, layer, loader=None):
    """TtRouter (replicated, fp32) for one layer, loading only router.{proj.weight, scale, per_expert_scale}."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, WeightLoader
    from models.demos.gemma4_a4b_d_p.tt.router import TtRouter

    loader = loader or WeightLoader(hf_path(spec))
    p = f"{PREFIX}layers.{layer}.router."
    return TtRouter(mesh, loader.get(p + "proj.weight"), loader.get(p + "scale"), loader.get(p + "per_expert_scale"))


def _router_host_fn(mesh, module):
    """fn(ctx, h_mid_host [S, H]) -> dense routing host [S, E]. h_mid goes up in fp32 (no bf16 rounding before top-8)."""
    import ttnn
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import replicated_to_host

    def fn(ctx, x):
        xd = ttnn.from_torch(
            x.float().reshape(1, 1, *x.shape[-2:]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        dense, idx, wts = module(xd)
        y = replicated_to_host(dense)
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _experts_module(mesh, spec, layer, loader=None):
    """TtExperts (EP=4, 32 experts per chip, fused unified_routed_expert_moe with GeluTanh) for one layer."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, WeightLoader
    from models.demos.gemma4_a4b_d_p.tt.experts import TtExperts

    loader = loader or WeightLoader(hf_path(spec))
    p = f"{PREFIX}layers.{layer}.experts."
    chunk = max([int(spec.get("target.chunk"))] + [int(r.get("chunk", 0)) for r in spec.data.get("ladder", [])])
    return TtExperts(mesh, layer, loader.get(p + "gate_up_proj"), loader.get(p + "down_proj"), max_seq_len=chunk)


def _experts_host_fn(mesh, module):
    """fn(ctx, moe_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H]."""
    import ttnn
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x, r):
        xd = to_device_replicated(mesh, x)
        rd = to_device_replicated(mesh, r)
        yd = module(xd, dense=rd)
        y = replicated_to_host(yd)
        for t in (xd, rd, yd):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _attention_module(mesh, spec, layer, loader=None, cfg=None):
    """TtSlidingAttention for a sliding layer, loading only its attention weights (not the experts)."""
    import os
    from types import SimpleNamespace

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, Gemma4TextConfig, WeightLoader, rope_inv_freq
    from models.demos.gemma4_a4b_d_p.tt.attention import TtSlidingAttention

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or Gemma4TextConfig.from_json(os.path.join(loader.model_path, "config.json"))
    if not cfg.is_sliding(layer):
        raise NotImplementedError(f"implement step: no device attention for global layer {layer} yet")
    p = f"{PREFIX}layers.{layer}.self_attn."
    names = {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm"}
    w = SimpleNamespace(**{k: loader.get(p + n + ".weight").float() for k, n in names.items()})
    inv_freq, _ = rope_inv_freq(cfg, True)
    return TtSlidingAttention(mesh, cfg, w, inv_freq, cfg.sliding_window, eps=cfg.rms_norm_eps), cfg


def _attention_host_fn(mesh, module, cfg, cache_of):
    """fn(ctx, x_host [S, H]) -> host [S, H]; cache_of(ctx) returns the layer's TtKVCacheSliding."""
    import ttnn
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd, ctx.start, cache_of(ctx))
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def device_component(mesh, spec, layer, step):
    if step in _NORM_WEIGHTS:
        return _host_fn(mesh, _norm_module(mesh, spec, layer, step))
    if step in _RESIDUAL_STEPS:
        return _residual_host_fn(mesh)
    if step == "mlp":
        return _host_fn(mesh, _mlp_module(mesh, spec, layer))
    if step == "router":
        return _router_host_fn(mesh, _router_module(mesh, spec, layer))
    if step == "experts":
        return _experts_host_fn(mesh, _experts_module(mesh, spec, layer))
    if step == "attention":
        from models.demos.gemma4_a4b_d_p.tt.attention import TtKVCacheSliding

        module, cfg = _attention_module(mesh, spec, layer)
        caches = {}

        def cache_of(ctx):
            # Component/swap tests: a fresh device cache holding the golden prefix [0, prefix_len).
            ex = ctx.extra or {}
            max_seq = int(ex.get("max_seq", ctx.start + ctx.length))
            max_seq = -(-max_seq // 32) * 32
            if "c" in caches:
                caches.pop("c").free()
            c = TtKVCacheSliding(mesh, cfg.num_key_value_heads, cfg.head_dim, max_seq)
            n = int(ex.get("prefix_len", ctx.start))
            if n:
                c.load_prefix(ex["state_prefix"]["key"], ex["state_prefix"]["value"], n)
            caches["c"] = c
            return c

        return _attention_host_fn(mesh, module, cfg, cache_of)
    raise NotImplementedError(f"implement step: no device module for {step} yet")


class _HybridState:
    """CPU reference state, except layers whose attention runs on the device: their K/V live in a device cache."""

    def __init__(self, ref, max_seq, mesh=None, device_attn_layers=(), cfg=None):
        from models.demos.gemma4_a4b_d_p.tt.attention import TtKVCacheSliding

        self.ref, self.s = ref, ref.new_state(max_seq)
        self.dev = {}
        if device_attn_layers:
            seq = -(-max_seq // 32) * 32
            self.dev = {
                i: TtKVCacheSliding(mesh, cfg.num_key_value_heads, cfg.head_dim, seq) for i in device_attn_layers
            }

    def load_prefix(self, layer, tensors, length):
        if layer in self.dev:
            self.dev[layer].load_prefix(tensors["key"], tensors["value"], length)
        else:
            self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        if layer in self.dev:
            return self.dev[layer].to_torch(length)
        return self.ref.state_tensors(self.s, layer, length)


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Hidden states stay on the host until more of the block is on the device."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import WeightLoader

        t0 = time.time()
        self.spec = spec
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.mesh = mesh
        self.overrides = {}
        self.attn_layers = []
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            ov = {s: _host_fn(mesh, _norm_module(mesh, spec, i, s, loader)) for s in steps if s in _NORM_WEIGHTS}
            ov.update({s: _residual_host_fn(mesh) for s in steps if s in _RESIDUAL_STEPS})
            if "mlp" in steps:
                ov["mlp"] = _host_fn(mesh, _mlp_module(mesh, spec, i, loader))
            if "router" in steps:
                ov["router"] = _router_host_fn(mesh, _router_module(mesh, spec, i, loader))
            if "experts" in steps:
                ov["experts"] = _experts_host_fn(mesh, _experts_module(mesh, spec, i, loader))
            if "attention" in steps:
                module, _ = _attention_module(mesh, spec, i, loader, self.cfg)
                ov["attention"] = _attention_host_fn(mesh, module, self.cfg, lambda ctx: ctx.extra["dev_cache"])
                self.attn_layers.append(i)
            self.overrides[i] = ov
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq, self.mesh, self.attn_layers, self.cfg)

    def embed(self, tokens):
        import torch.nn.functional as F

        return F.embedding(tokens.long(), self.ref.embed) * self.ref.embed_scale

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        if i in state.dev:
            ctx = Ctx(ctx.layer, ctx.start, ctx.length, ctx.state, {**ctx.extra, "dev_cache": state.dev[i]})
        return run_block(
            self.ref.block_graph(i),
            lambda n: self.ref.component(i, n),
            ctx,
            h,
            overrides=self.overrides[i],
        )

    def final_norm(self, h):
        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import rms_norm

        return rms_norm(h, self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
