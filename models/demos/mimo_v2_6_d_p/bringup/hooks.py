# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v2_6_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def _physical_threads():
    """Use one torch thread per physical core: the gates set os.cpu_count() (SMT) threads, which makes the many small
    per-expert GEMMs and mxfp4 dequants 2-7x slower on this host."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    if n and torch.get_num_threads() > n:
        torch.set_num_threads(n)


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/mimo_ref.py). Experts are dequantized at load for small layer sets
    (the 0-5 subset) and kept mxfp4, dequantized per use, for the 48-layer parity run."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoReference

    _physical_threads()
    return MiMoReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF oracle: the checkpoint's modeling_mimo_v2.py, text decoder only, fp32, eager attention, weights
    dequantized as the checkpoint defines them, routed experts kept mxfp4 and dequantized when they run
    (reference/hf_oracle.py). num_layers=None is the full 48-layer model (HF sanity gate)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.hf_oracle import build_hf_model

    _physical_threads()
    return build_hf_model(hf_path(spec), num_layers, torch.float32)


# Device steps swapped in so far, per block type. Every other step runs on the CPU reference.
DEVICE_STEPS = {
    "full_dense": {"attn_norm", "attention", "attn_residual", "mlp", "mlp_residual"},
    "sliding_moe": {"attn_norm", "attention", "router", "experts"},
}

# Residual steps (replicated bf16 add, no collective): h_mid = in + attn_out; out = h_mid + mlp_out.
_RESIDUAL_STEPS = {"attn_residual", "mlp_residual"}

# Norm steps -> checkpoint weight name (under model.layers.<i>.).
_NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def _norm_module(mesh, spec, layer, step, loader=None):
    """TtRMSNorm (replicated, HiFi4 + fp32 acc, plain w) for one layer's norm step; loads only that weight."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import TtRMSNorm

    loader = loader or WeightLoader(hf_path(spec))
    w = loader.get(f"model.layers.{layer}.{_NORM_WEIGHTS[step]}")
    return TtRMSNorm(mesh, w, eps=1e-6)


def _host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H] (component/swap/hybrid harness boundary)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd)
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def _residual_host_fn(mesh):
    """fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H] via TtResidualAdd (a + b on the device)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p.tt.residual import TtResidualAdd
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

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
    """TtDenseMLP (TP=4 SwiGLU) for the dense layer; fp8 + 128x128 block scale dequantized to bf16 at load."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader, fp8_weight
    from models.demos.mimo_v2_6_d_p.tt.mlp import TtDenseMLP

    loader = loader or WeightLoader(hf_path(spec))
    p = f"model.layers.{layer}.mlp."
    wg, wu, wd = (fp8_weight(loader, p + f"{n}.weight", torch.float32) for n in ("gate_proj", "up_proj", "down_proj"))
    return TtDenseMLP(mesh, wg, wu, wd)


def _rope_max_seq(spec):
    """Longest sequence any rung or the target runs: the RoPE tables are built once for it at load."""
    seqs = [int(r.get("seq", 0)) for r in spec.data.get("ladder", [])]
    seqs.append(int((spec.data.get("target") or {}).get("seq", 0)))
    return max(seqs)


def _attention_module(mesh, spec, layer, loader=None, cfg=None):
    """TtFullAttention (full layers) or TtSlidingAttention (sliding layers: window, per-head sink) for one layer, loading
    only its attention weights (fused qkv dequantized per TP rank, bf16 o_proj, bf16 sink bias)."""
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig, rope_inv_freq
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader, qkv_weight
    from models.demos.mimo_v2_6_d_p.tt.attention import TtFullAttention, TtSlidingAttention

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))
    hq, hkv, d, dv = cfg.attn_dims(layer)
    p = f"model.layers.{layer}.self_attn."
    wqkv = qkv_weight(loader, p, (hq * d, hkv * d, hkv * dv), torch.float32)
    wo = loader.get(p + "o_proj.weight").float()
    sliding = cfg.is_sliding(layer)
    inv_freq = rope_inv_freq(cfg.swa_rope_theta if sliding else cfg.rope_theta, cfg.rope_dim(layer))
    dims, max_seq, vs = (hq, hkv, d, dv), _rope_max_seq(spec), cfg.attention_value_scale
    if sliding:
        sink = loader.get(p + "attention_sink_bias").float() if cfg.has_sink(layer) else None
        module = TtSlidingAttention(mesh, wqkv, wo, dims, inv_freq, max_seq, vs, cfg.sliding_window, sink)
    else:
        assert not cfg.has_sink(layer)
        module = TtFullAttention(mesh, wqkv, wo, dims, inv_freq, max_seq, vs)
    return module, cfg


def _max_chunk(spec):
    """Largest chunk any rung or the target runs: the router's bias / zero tables are built once for it at load."""
    chunks = [int(r.get("chunk", 0)) for r in spec.data.get("ladder", [])]
    chunks.append(int((spec.data.get("target") or {}).get("chunk", 0)))
    return max(chunks)


def _router_module(mesh, spec, layer, loader=None, cfg=None):
    """TtRouter (replicated, fp32 logits, fp32 sigmoid + bias choice, ttnn.topk) for one MoE layer."""
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p.tt.router import TtRouter

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))
    assert cfg.n_group == 1 and cfg.scoring_func == "sigmoid" and cfg.norm_topk_prob
    p = f"model.layers.{layer}.mlp.gate."
    w = loader.get(p + "weight").float()
    b = loader.get(p + "e_score_correction_bias").float()
    rs = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
    mode = os.environ.get("MIMO_ROUTER_MODE", "fp32")  # "fused": moe_grouped_topk (TF32 keys), for comparison
    return TtRouter(mesh, w, b, _max_chunk(spec), top_k=cfg.num_experts_per_tok, route_scale=rs, mode=mode)


def _router_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> dense routing host [S, E] (chip 0's copy of the replicated result)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        dense, idx, wts = module(xd)
        y = replicated_to_host(dense)
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _experts_module(mesh, spec, layer, loader=None, cfg=None):
    """TtExperts (EP=4, 64 experts per chip, bfp8, local dispatch -> per-expert SwiGLU -> combine) for one MoE layer. The mxfp4
    experts are dequantized one at a time at load unless the bfp8 device cache for the layer is complete."""
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p.tt.experts import LazyExpertWeights, TtExperts

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))
    weights = LazyExpertWeights(loader, f"model.layers.{layer}.mlp.experts.", cfg.n_routed_experts)
    # "loop" (default): per-expert extract -> ttnn.linear SwiGLU (HiFi2, fp32 dest) -> insert; "unified":
    # unified_routed_expert_moe (Silu forced to LoFi + bf16 dest, fails the norm-ratio check); "fused": moe_fused_swiglu.
    mode = os.environ.get("MIMO_EXPERTS_MODE", "loop")
    return TtExperts(
        mesh,
        layer,
        weights,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        max_seq_len=_max_chunk(spec),
        mode=mode,
    )


def _experts_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H]."""
    import ttnn
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x, r):
        xd = to_device_replicated(mesh, x)
        rd = to_device_replicated(mesh, r)
        yd = module(xd, dense=rd)
        y = replicated_to_host(yd)
        for t in (xd, rd, yd):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _new_kv_cache(mesh, cfg, layer, max_seq):
    """Empty device KV cache for one layer, V padded to 192: full layers 4 KV heads (head r on chip r, paged-shaped),
    sliding layers 8 KV heads (heads 2r, 2r+1 on chip r, contiguous)."""
    from models.demos.mimo_v2_6_d_p.tt.attention import TtKVCacheFull, TtKVCacheSliding

    _, hkv, d, dv = cfg.attn_dims(layer)
    if cfg.is_sliding(layer):
        return TtKVCacheSliding(mesh, hkv, d, dv, max_seq)
    return TtKVCacheFull(mesh, hkv, d, dv, max_seq)


def _attention_host_fn(mesh, module, cache_of):
    """fn(ctx, x_host [S, H]) -> host [S, H]; cache_of(ctx) returns the layer's device KV cache."""
    import ttnn
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

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
    if step == "attention":
        module, cfg = _attention_module(mesh, spec, layer)
        caches = {}

        def cache_of(ctx):
            # Component/swap tests: a fresh device cache holding the golden prefix [0, prefix_len).
            ex = ctx.extra or {}
            max_seq = int(ex.get("max_seq", ctx.start + ctx.length))
            if "c" in caches:
                caches.pop("c").free()
            c = _new_kv_cache(mesh, cfg, layer, max_seq)
            n = int(ex.get("prefix_len", ctx.start))
            if n:
                c.load_prefix(ex["state_prefix"]["key"], ex["state_prefix"]["value"], n)
            caches["c"] = c
            return c

        return _attention_host_fn(mesh, module, cache_of)
    if step in _RESIDUAL_STEPS:
        return _residual_host_fn(mesh)
    if step == "mlp":
        return _host_fn(mesh, _mlp_module(mesh, spec, layer))
    if step == "router":
        return _router_host_fn(mesh, _router_module(mesh, spec, layer))
    if step == "experts":
        return _experts_host_fn(mesh, _experts_module(mesh, spec, layer))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


class _HybridState:
    """CPU reference state, except layers whose attention runs on the device: their K/V live in a device cache."""

    def __init__(self, ref, max_seq, mesh=None, device_attn_layers=(), cfg=None):
        self.ref, self.s = ref, ref.new_state(max_seq)
        self.dev = {i: _new_kv_cache(mesh, cfg, i, max_seq) for i in device_attn_layers}

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
    Hidden states stay on the host until the assemble step builds the all-device model."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

        t0 = time.time()
        self.spec = spec
        self.mesh = mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.overrides = {}
        self.attn_layers = []
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            ov = {s: _host_fn(mesh, _norm_module(mesh, spec, i, s, loader)) for s in steps if s in _NORM_WEIGHTS}
            ov.update({s: _residual_host_fn(mesh) for s in steps if s in _RESIDUAL_STEPS})
            if "mlp" in steps:
                ov["mlp"] = _host_fn(mesh, _mlp_module(mesh, spec, i, loader))
            if "router" in steps:
                ov["router"] = _router_host_fn(mesh, _router_module(mesh, spec, i, loader, self.cfg))
            if "experts" in steps:
                ov["experts"] = _experts_host_fn(mesh, _experts_module(mesh, spec, i, loader, self.cfg))
            if "attention" in steps:
                module, _ = _attention_module(mesh, spec, i, loader, self.cfg)
                ov["attention"] = _attention_host_fn(mesh, module, lambda ctx: ctx.extra["dev_cache"])
                self.attn_layers.append(i)
            self.overrides[i] = ov
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq, self.mesh, self.attn_layers, self.cfg)

    def embed(self, tokens):
        import torch.nn.functional as F

        return F.embedding(tokens.long(), self.ref.embed)

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
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rms_norm

        return rms_norm(h, self.ref.final_norm_w, self.cfg.layernorm_epsilon)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """Hybrid for now (CPU reference + DEVICE_STEPS on the device); the assemble step replaces it."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
