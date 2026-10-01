# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v2_6_d_p_cp4: the same checkpoint as the prior bring-up mimo_v2_6_d_p (spec ``prior``), on another mesh.

CPU side: the prior's hooks (same reference, goldens and HF loader); change them there only for a bug.
Device side (implement role): device_component, device_model, written for this bring-up's plan.
"""

from models.demos.mimo_v2_6_d_p.bringup import hooks as _prior

reference = _prior.reference
for _name in ("tokenizer", "hf_model", "hf_layers"):
    if hasattr(_prior, _name):
        globals()[_name] = getattr(_prior, _name)


# Norm steps -> checkpoint weight name (under model.layers.<i>.).
_NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def _norm_module(mesh, spec, layer, step, loader=None):
    """TtRMSNorm (CP slice per chip, replicated weight, HiFi4 + fp32 acc, plain w); loads only that weight."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import TtRMSNorm

    loader = loader or WeightLoader(hf_path(spec))
    return TtRMSNorm(mesh, loader.get(f"model.layers.{layer}.{_NORM_WEIGHTS[step]}"), eps=1e-6)


def _mlp_module(mesh, spec, layer, loader=None):
    """TtDenseMLP (TP=1, local to each CP slice) for the dense layer; fp8 + 128x128 block scale dequantized at load."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p_cp4.tt.mlp import build_mlp

    return build_mlp(mesh, loader or WeightLoader(hf_path(spec)), layer)


def _cp_host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H]: the input is split into the 4 CP slices
    (chip c gets rows [c*S/4, (c+1)*S/4)) and the outputs are concatenated back (component/swap harness boundary)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import cp_to_host, to_device_cp

    def fn(ctx, x):
        xd = to_device_cp(mesh, x)
        yd = module(xd)
        y = cp_to_host(mesh, yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


# Residual steps (bf16 add of two CP slices of the same rows, no collective).
_RESIDUAL_STEPS = {"attn_residual", "mlp_residual", "ffn_residual"}


def _residual_host_fn(mesh):
    """fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H] via TtResidualAdd on the CP slices."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_cp4.tt.residual import TtResidualAdd
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import cp_to_host, to_device_cp

    module = TtResidualAdd(mesh)

    def fn(ctx, a, b):
        ad = to_device_cp(mesh, a)
        bd = to_device_cp(mesh, b)
        yd = module(ad, bd)
        y = cp_to_host(mesh, yd)
        for t in (ad, bd, yd):
            ttnn.deallocate(t)
        return y.to(a.dtype)

    return fn


def _rope_max_seq(spec):
    """Longest sequence any rung or the target runs: the RoPE tables are built once for it at load."""
    seqs = [int(r.get("seq", 0)) for r in spec.data.get("ladder", [])]
    seqs.append(int((spec.data.get("target") or {}).get("seq", 0)))
    return max(seqs)


def _chunk_sizes(spec):
    """Every chunk size a rung or the target runs: the chunk-major RoPE tables are built once for each at load."""
    chunks = {int(r["chunk"]) for r in spec.data.get("ladder", []) if r.get("chunk")}
    if (spec.data.get("target") or {}).get("chunk"):
        chunks.add(int(spec.data["target"]["chunk"]))
    return sorted(chunks)


def _attention_module(mesh, spec, layer, ccl, loader=None, cfg=None):
    """TtFullAttention / TtSlidingAttention (CP=4 ring, TP=1) for one layer, loading only its attention weights
    (fused qkv dequantized per stored TP-rank slab and reassembled in global order, bf16 o_proj, bf16 sink)."""
    import os

    import torch

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig, rope_inv_freq
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader, qkv_weight
    from models.demos.mimo_v2_6_d_p_cp4.tt.attention import TtFullAttention, TtSlidingAttention

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))
    sliding = cfg.is_sliding(layer)
    hq, hkv, d, dv = cfg.attn_dims(layer)
    p = f"model.layers.{layer}.self_attn."
    wqkv = qkv_weight(loader, p, (hq * d, hkv * d, hkv * dv), torch.float32)
    wo = loader.get(p + "o_proj.weight").float()
    inv_freq = rope_inv_freq(cfg.swa_rope_theta if sliding else cfg.rope_theta, cfg.rope_dim(layer))
    common = (mesh, ccl, wqkv, wo, (hq, hkv, d, dv), inv_freq, _rope_max_seq(spec), cfg.attention_value_scale)
    if sliding:
        sink = loader.get(p + "attention_sink_bias").float() if cfg.has_sink(layer) else None
        module = TtSlidingAttention(*common, _chunk_sizes(spec), cfg.sliding_window, sink)
    else:
        assert not cfg.has_sink(layer)
        module = TtFullAttention(*common, _chunk_sizes(spec))
    return module, cfg


def _attention_host_fn(mesh, module, cache_of):
    """fn(ctx, x_host [S, H]) -> host [S, H] (CP slices in and out); cache_of(ctx) returns the layer's ring cache."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import cp_to_host, to_device_cp

    def fn(ctx, x):
        cache = cache_of(ctx)
        cache.bind_chunk(ctx.length)  # harness boundary: writes a prefix loaded before the chunk size was known
        xd = to_device_cp(mesh, x)
        yd = module(xd, ctx.start, cache)
        y = cp_to_host(mesh, yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def _max_chunk(spec):
    """Largest chunk any rung or the target runs."""
    chunks = [int(r.get("chunk", 0)) for r in spec.data.get("ladder", [])]
    chunks.append(int((spec.data.get("target") or {}).get("chunk", 0)))
    return max(chunks)


def _router_module(mesh, spec, layer, loader=None, cfg=None):
    """TtRouter (replicated weight, each chip routes its own S/4 CP rows, fp32 logits, fp32 sigmoid + bias choice,
    ttnn.topk). Zero/bias tables are built once for the largest per-chip slice (max chunk / 4).
    MIMO_ROUTER_MODE=fused selects moe_grouped_topk (TF32 keys) for comparison."""
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p_cp4.tt.router import TtRouter

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))
    assert cfg.n_group == 1 and cfg.scoring_func == "sigmoid" and cfg.norm_topk_prob
    p = f"model.layers.{layer}.mlp.gate."
    w = loader.get(p + "weight").float()
    b = loader.get(p + "e_score_correction_bias").float()
    rs = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
    rows = -(-_max_chunk(spec) // mesh.get_num_devices())
    mode = os.environ.get("MIMO_ROUTER_MODE", "fp32")
    return TtRouter(mesh, w, b, rows, top_k=cfg.num_experts_per_tok, route_scale=rs, mode=mode)


def _router_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> dense routing host [S, E]: CP slices in, the 4 per-chip [S/4, E] results concatenated."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import cp_to_host, to_device_cp

    def fn(ctx, x):
        xd = to_device_cp(mesh, x)
        dense, idx, wts = module(xd)
        y = cp_to_host(mesh, dense)
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _experts_module(mesh, spec, layer, loader=None, cfg=None):
    """TtExperts (CP=4 EP=4: each chip dispatches its own S/4 rows over one 4-chip group on axis 1, 64 complete
    experts per chip, unified_routed_expert_moe high_precision at HiFi4, combine, post_combine_reduce; no CCL after)."""
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p_cp4.tt.experts import build_experts

    loader = loader or WeightLoader(hf_path(spec))
    cfg = cfg or MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))
    return build_experts(mesh, loader, cfg, layer, _max_chunk(spec))


def _experts_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H] (CP slices in and out)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import cp_to_host, to_device_cp

    def fn(ctx, x, r):
        xd = to_device_cp(mesh, x)
        rd = to_device_cp(mesh, r)
        yd = module(xd, dense=rd)
        y = cp_to_host(mesh, yd)
        for t in (xd, rd, yd):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def device_component(mesh, spec, layer, step):
    if step in _NORM_WEIGHTS:
        return _cp_host_fn(mesh, _norm_module(mesh, spec, layer, step))
    if step in _RESIDUAL_STEPS:
        return _residual_host_fn(mesh)
    if step == "mlp":
        return _cp_host_fn(mesh, _mlp_module(mesh, spec, layer))
    if step == "router":
        return _router_host_fn(mesh, _router_module(mesh, spec, layer))
    if step == "experts":
        return _experts_host_fn(mesh, _experts_module(mesh, spec, layer))
    if step == "attention":
        from models.demos.mimo_v2_6_d_p_cp4.tt.ccl import RingCCL

        module, cfg = _attention_module(mesh, spec, layer, RingCCL(mesh))
        caches = {}

        def cache_of(ctx):
            # Component/swap tests: a fresh ring cache (laid out for this chunk) holding the golden prefix.
            ex = ctx.extra or {}
            max_seq = int(ex.get("max_seq", ctx.start + ctx.length))
            if "c" in caches:
                caches.pop("c").free()
            c = module.new_cache(max_seq, chunk=ctx.length)
            n = int(ex.get("prefix_len", ctx.start))
            if n:
                c.load_prefix(ex["state_prefix"]["key"], ex["state_prefix"]["value"], n)
            caches["c"] = c
            return c

        return _attention_host_fn(mesh, module, cache_of)
    raise NotImplementedError(f"implement step: no device module for {step} yet")


# Device steps of the hybrid harness, per block type: each passed its component gate on the device (CP slices).
# Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "full_dense": {"attn_norm", "attention", "attn_residual", "mlp"},
    "sliding_moe": {"attn_norm", "attention", "router", "experts"},
    "full_moe": set(),
}


class _HybridState:
    """CPU reference state, except layers whose attention runs on the device: their K/V live in a device ring cache
    (laid out chunk-major for the chunk size of the first chunk run; a prefix loaded before is written then)."""

    def __init__(self, ref, max_seq, attn_modules=None):
        self.ref, self.s = ref, ref.new_state(max_seq)
        self.dev = {i: m.new_cache(max_seq) for i, m in (attn_modules or {}).items()}

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
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step,
    the input split into the 4 CP slices). Hidden states stay on the host until the assemble step."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        import torch

        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.overrides, self.attn = {}, {}
        self.ccl = None
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            ov = {s: _cp_host_fn(mesh, _norm_module(mesh, spec, i, s, loader)) for s in steps if s in _NORM_WEIGHTS}
            ov.update({s: _residual_host_fn(mesh) for s in steps if s in _RESIDUAL_STEPS})
            if "mlp" in steps:
                ov["mlp"] = _cp_host_fn(mesh, _mlp_module(mesh, spec, i, loader))
            if "router" in steps:
                ov["router"] = _router_host_fn(mesh, _router_module(mesh, spec, i, loader, self.cfg))
            if "experts" in steps:
                ov["experts"] = _experts_host_fn(mesh, _experts_module(mesh, spec, i, loader, self.cfg))
            if "attention" in steps:
                from models.demos.mimo_v2_6_d_p_cp4.tt.ccl import RingCCL

                self.ccl = self.ccl or RingCCL(mesh)
                self.attn[i], _ = _attention_module(mesh, spec, i, self.ccl, loader, self.cfg)
                ov["attention"] = _attention_host_fn(mesh, self.attn[i], lambda ctx: ctx.extra["dev_cache"])
            self.overrides[i] = ov
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq, self.attn)

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
    """The hybrid harness (CPU reference + DEVICE_STEPS on the device) until the assemble step adds the all-device
    model."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
