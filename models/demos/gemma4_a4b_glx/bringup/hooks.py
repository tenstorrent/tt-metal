# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gemma4_a4b_glx: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

PREFIX = "model.language_model."

_NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "post_attn_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight",
    "post_mlp_norm": "post_feedforward_layernorm_1.weight",
    "moe_norm": "pre_feedforward_layernorm_2.weight",
    "post_moe_norm": "post_feedforward_layernorm_2.weight",
    "post_ffn_norm": "post_feedforward_layernorm.weight",
}


def _weight_loader(spec):
    from models.demos.gemma4_a4b_glx.reference.gemma4_ref import WeightLoader

    return WeightLoader(str(spec.hf_dir))


def _norm_module(mesh, spec, layer, step):
    from models.demos.gemma4_a4b_glx.tt.rms_norm import TtRMSNorm

    loader = _weight_loader(spec)
    w = loader.get(f"{PREFIX}layers.{layer}.{_NORM_WEIGHTS[step]}")
    return TtRMSNorm(mesh, w, eps=1e-6)


def _host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H]."""
    import ttnn
    from models.demos.gemma4_a4b_glx.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd)
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def _host_fn2(mesh, module):
    """Wrap a two-input device module as fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H]."""
    import ttnn
    from models.demos.gemma4_a4b_glx.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, a, b):
        ad = to_device_replicated(mesh, a)
        bd = to_device_replicated(mesh, b)
        yd = module(ad, bd)
        y = replicated_to_host(yd)
        for t in (ad, bd, yd):
            ttnn.deallocate(t)
        return y.to(a.dtype)

    return fn


def _attention_module(mesh, spec, layer):
    """Sliding attention of `layer`; loads only that layer's six self_attn tensors."""
    import os

    from models.demos.gemma4_a4b_glx.reference.gemma4_ref import Gemma4TextConfig, rope_inv_freq
    from models.demos.gemma4_a4b_glx.tt.attention import TtSlidingAttention

    cfg = Gemma4TextConfig.from_json(os.path.join(str(spec.hf_dir), "config.json"))
    if not cfg.is_sliding(layer):
        raise NotImplementedError(f"implement stage: no device module for global attention (layer {layer}) yet")
    loader = _weight_loader(spec)
    p = f"{PREFIX}layers.{layer}.self_attn."
    inv_freq, _ = rope_inv_freq(cfg, sliding=True)
    return TtSlidingAttention(
        mesh,
        loader.get(p + "q_proj.weight"),
        loader.get(p + "k_proj.weight"),
        loader.get(p + "v_proj.weight"),
        loader.get(p + "o_proj.weight"),
        loader.get(p + "q_norm.weight"),
        loader.get(p + "k_norm.weight"),
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        window=cfg.sliding_window,
        rope_inv_freq=inv_freq,
        eps=cfg.rms_norm_eps,
        scale=1.0,
    )


def _attention_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host attn_out [S, H]. The K/V prefix comes from ctx.extra["state_prefix"]; a fresh
    device cache is built from it on each call (the component harness gives the state as host tensors)."""
    import ttnn
    from models.demos.gemma4_a4b_glx.tt.attention import TtKVCache
    from models.demos.gemma4_a4b_glx.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        extra = ctx.extra
        prefix = extra.get("state_prefix") or {}
        max_seq = int(extra.get("max_seq", ctx.start + x.shape[0]))
        n = int(extra.get("prefix_len", ctx.start))
        assert n == ctx.start, f"prefix_len {n} != chunk start {ctx.start}"
        k_pre, v_pre = (prefix[name][:, :n] if name in prefix else None for name in ("key", "value"))
        module.ensure_rope(max_seq)
        cache = TtKVCache(mesh, module.nkv * mesh.shape[1], module.d, max_seq, k_pre, v_pre)
        xd = to_device_replicated(mesh, x)
        yd = module(xd, cache, ctx.start)
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        cache.deallocate()
        return y.to(x.dtype)

    return fn


def _router_module(mesh, spec, layer):
    """Router of `layer`; loads only router.{proj.weight, scale, per_expert_scale}."""
    import os

    from models.demos.gemma4_a4b_glx.reference.gemma4_ref import Gemma4TextConfig
    from models.demos.gemma4_a4b_glx.tt.router import TtRouter

    cfg = Gemma4TextConfig.from_json(os.path.join(str(spec.hf_dir), "config.json"))
    loader = _weight_loader(spec)
    p = f"{PREFIX}layers.{layer}.router."
    return TtRouter(
        mesh,
        loader.get(p + "proj.weight"),
        loader.get(p + "scale"),
        loader.get(p + "per_expert_scale"),
        top_k=cfg.top_k_experts,
        eps=cfg.rms_norm_eps,
    )


def _router_fn(mesh, module):
    """fn(ctx, h_mid_host [S, H]) -> host dense routing [S, E] (fp32). h_mid goes up as bf16, as the device
    residual stream is (the golden is bf16, so this is exact on the golden)."""
    import ttnn
    from models.demos.gemma4_a4b_glx.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        dense, ids = module(xd)
        y = replicated_to_host(dense)
        for t in (xd, dense, ids):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _experts_fn(mesh, spec, layer):
    """fn(ctx, moe_norm_host [S, H], router_host [S, E]) -> host experts_out [S, H]. Loads only the layer's
    experts.{gate_up_proj, down_proj}; the dense routing goes up as fp32, as the device router produces it."""
    import ttnn
    from models.demos.gemma4_a4b_glx.tt.experts import TtExperts
    from models.demos.gemma4_a4b_glx.tt.rms_norm import replicated_to_host, to_device_replicated

    loader = _weight_loader(spec)
    p = f"{PREFIX}layers.{layer}.experts."
    module = TtExperts(mesh, loader.get(p + "gate_up_proj"), loader.get(p + "down_proj"))

    def fn(ctx, x, router):
        xd = to_device_replicated(mesh, x)
        rd = ttnn.from_torch(
            router.float().reshape(1, 1, *router.shape[-2:]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        yd = module(xd, rd)
        y = replicated_to_host(yd)
        for t in (xd, rd, yd):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def device_component(mesh, spec, layer, step):
    import ttnn

    # FastOperation bypasses PRE_OPERATION_HOOKS; DeviceProof needs the slow path to count ops.
    ttnn.CONFIG.enable_fast_runtime_mode = False
    if step in _NORM_WEIGHTS:
        return _host_fn(mesh, _norm_module(mesh, spec, layer, step))
    if step == "attention":
        return _attention_fn(mesh, _attention_module(mesh, spec, layer))
    if step == "mlp":
        from models.demos.gemma4_a4b_glx.tt.mlp import TtMLP

        loader = _weight_loader(spec)
        p = f"{PREFIX}layers.{layer}.mlp."
        mlp = TtMLP(mesh, *(loader.get(p + f"{n}_proj.weight") for n in ("gate", "up", "down")))
        return _host_fn(mesh, mlp)
    if step == "attn_residual":
        from models.demos.gemma4_a4b_glx.tt.residual import TtResidualAdd

        return _host_fn2(mesh, TtResidualAdd(mesh))
    if step == "router":
        return _router_fn(mesh, _router_module(mesh, spec, layer))
    if step == "experts":
        return _experts_fn(mesh, spec, layer)
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
