# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-side hooks for mistral_large_4: per-step device modules (device_component) and the all-device model
(device_model). The CPU side is in reference/mistral_large_4/hooks.py; the block graph in
reference/mistral_large_4/_bringup_interface.py.
"""

import types

_WEIGHTS = {}  # (hf dir, layer) -> {key: tensor}: the layer's dense weights, read once per process


def _layer_weights(spec, layer: int, keys) -> dict:
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4._bringup_interface import weight_dtype
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config, layer_names
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, load_dense

    cache = _WEIGHTS.setdefault((str(spec.hf_dir), layer), {})
    missing = [k for k in keys if k not in cache]
    if missing:
        names = layer_names(Config(spec.hf_dir), layer)
        cache.update(load_dense(checkpoint(spec.hf_dir), {k: names[k] for k in missing}, weight_dtype(spec)))
    return {k: cache[k] for k in keys}


def _norm(mesh, spec, layer: int, key: str):
    import ttnn
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

    # attn_norm feeds the indexer, which reads it at ATTN_NORM_DTYPE
    dtype = getattr(ttnn, settings.get("ATTN_NORM_DTYPE")) if key == "attn_norm" else ttnn.bfloat16
    return TtDistributedRmsNorm(mesh, _layer_weights(spec, layer, [key])[key], Config(spec.hf_dir).eps, out_dtype=dtype)


def _counted(module, *x):
    """module(*x) on ttnn's standard Operation path, so pre-operation hooks see its ops (FastOperation skips them);
    the ops are the same. Scoped to this call."""
    import ttnn

    fast = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    try:
        return module(*x)
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = fast


_MODULES = {}  # (mesh id, layer, step) -> device module, built once per process


def _indexer(mesh, spec, layer: int):
    key = (id(mesh), layer, "indexer")
    if key not in _MODULES:
        _MODULES[key] = _new_indexer(mesh, spec, layer)
    return _MODULES[key]


def _inv_freq(spec, cfg, layer: int):
    """The layer's RoPE inverse frequencies (None on NoPE layers): YaRN x LeRoPE exp(alpha), as the reference."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4._bringup_interface import weight_dtype
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import yarn_inv_freq
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, load_dense

    if not cfg.use_rope(layer):
        return None
    alpha = load_dense(checkpoint(spec.hf_dir), {"a": "lerope.alpha"}, weight_dtype(spec))["a"].float()
    return yarn_inv_freq(cfg.rope, cfg.qk_rope) * alpha.exp()


def _seq(spec):
    return int(spec.get("target.seq") or spec.get("shapes.max_seq_len")), int(
        spec.get("target.chunk") or spec.get("shapes.chunk_size")
    )


def _new_indexer(mesh, spec, layer: int, **cache):
    """cache: TtIndexer's cache, cache_layer, cache_layers (the served index cache and this layer's region); without
    them the module allocates a cache of its own."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config, layer_names
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.indexer import TtIndexer

    cfg = Config(spec.hf_dir)
    keys = [k for k in layer_names(cfg, layer) if k.startswith("idx_")]
    max_seq, chunk = _seq(spec)
    return TtIndexer(
        mesh,
        _layer_weights(spec, layer, keys),
        inv_freq=_inv_freq(spec, cfg, layer),
        n_heads=cfg.index_heads,
        head_dim=cfg.index_dim,
        rope_dim=cfg.rope_dim(layer),
        topk=cfg.index_topk,
        max_seq=max_seq,
        chunk=chunk,
        **cache,
    )


def _attention(mesh, spec, layer: int):
    key = (id(mesh), layer, "attention")
    if key not in _MODULES:
        _MODULES[key] = _new_attention(mesh, spec, layer)
    return _MODULES[key]


def _new_attention(mesh, spec, layer: int, **cache):
    """cache: TtAttention's cache, cache_layer, cache_layers (the served kvpe cache and the region the layer owns, or
    reads on a shared-KV layer); without them the module allocates a cache of its own."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config, layer_names
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import TtAttention

    cfg = Config(spec.hf_dir)
    # a shared-KV layer has no wkv_a / kv_a_norm: it reads its source layer's kvpe region
    keys = ["wq", "wkv_b", "wo", "ssmax_s", "ssmax_b"] + (["wkv_a", "kv_a_norm"] if cfg.owns_kv(layer) else [])
    assert all(k in layer_names(cfg, layer) for k in keys)
    max_seq, chunk = _seq(spec)
    return TtAttention(
        mesh,
        _layer_weights(spec, layer, keys),
        inv_freq=_inv_freq(spec, cfg, layer),
        n_heads=cfg.n_heads,
        nope_dim=cfg.nope_dim(layer),
        rope_dim=cfg.rope_dim(layer),
        v_dim=cfg.v_head,
        kv_lora=cfg.kv_lora,
        eps=cfg.eps,
        max_seq=max_seq,
        chunk=chunk,
        pe_dim=cfg.qk_rope,  # NoPE layers keep the 576-wide kvpe record, pe zero
        **cache,
    )


def _experts(mesh, spec, layer: int):
    import torch

    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, dequant, expert_store
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.experts import TtExperts
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS

    key = (id(mesh), layer, "experts")
    if key not in _MODULES:
        cfg = Config(spec.hf_dir)
        le = expert_store(checkpoint(spec.hf_dir), cfg.n_experts).get(layer)
        chunk = int(spec.get("target.chunk") or spec.get("shapes.chunk_size"))
        _MODULES[key] = TtExperts(
            mesh,
            lambda e, m: dequant(le.w[(e, m)], le.s[(e, m)], torch.bfloat16),  # FP8 x block scale: exact in bf16
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            rows_per_chip=chunk // mesh.shape[SP_AXIS],
        )
    return _MODULES[key]


def swap_context(spec, ref, layer, golden, chunk, rctx, dctx):
    """The reference hooks' swap_context (the golden's source-layer top-k / KV), with the reference state's KV prefix
    rows [0, chunk start) rounded through bfloat8_b as the device's bfp8 kvpe cache holds them, so the CPU step reads
    the same values the device attends:

      * a shared-KV layer's source prefix rows (``shared_kv``) are rounded and also put on the reference state
        (``shared_kv_prefix``, views of the same tensors); the chunk's own rows stay full precision (ATTN_CHUNK_KV).
      * an owning layer's own kv / kr prefix rows are rounded in place.

    Index keys stay exact."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4 import hooks as reference_hooks

    reference_hooks.swap_context(spec, ref, layer, golden, chunk, rctx, dctx)
    src = rctx.extra.get("shared_kv")
    if src is not None and rctx.start:
        # new tensors: the device context keeps the golden's (it writes them into its bfp8 cache itself)
        import torch

        s = rctx.start
        src = {n: torch.cat((_served_kv(t[:s]), t[s:])) if t.numel() else t for n, t in src.items()}
        rctx.extra["shared_kv"] = src
    if src is not None and rctx.state is not None:
        rctx.state.shared_kv_prefix = {n: t[: rctx.start] for n, t in src.items() if t.numel()}
    if rctx.state is not None and rctx.start:
        for cache in (rctx.state.kv, rctx.state.kr):
            t = cache.get(layer)
            if t is not None and t.numel():
                t[: rctx.start] = _served_kv(t[: rctx.start])


def _served_kv(t):
    """Host rows [n, d] rounded through bfloat8_b, as the served kvpe cache holds them: blocks of 16 values along a row
    share the largest exponent, 7 mantissa bits each, round to nearest."""
    import torch

    n, d = t.shape
    blk = t.float().reshape(n, d // 16, 16)
    amax = blk.abs().amax(-1, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(amax.clamp_min(1e-38))) - 6)
    q = (torch.round(blk / step).clamp(-127, 127) * step).where(amax > 0, torch.zeros_like(blk))
    return q.reshape(n, d).to(t.dtype)


def device_component(mesh, spec, layer, step):
    """fn(ctx, *host inputs) -> host tensor. The host <-> device conversion (layout.py) wraps a module that runs on
    the device only."""
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import residual_to_device, residual_to_host

    if step in ("attn_norm", "ffn_norm"):
        mod = _norm(mesh, spec, layer, step)

        def fn(ctx, x):
            return residual_to_host(_counted(mod, residual_to_device(x, mesh)), mesh)

        return fn
    if step == "indexer":
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import index_cache_load, topk_to_host
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

        mod = _indexer(mesh, spec, layer)
        x_dtype = getattr(ttnn, settings.get("ATTN_NORM_DTYPE"))  # attn_norm's output dtype, as the device hands it on

        def fn(ctx, x):
            start = ctx.extra.get("prefix_len", ctx.start)
            index_cache_load(mod, ctx.extra.get("state_prefix", {}).get("index_key"), start, mesh)
            return topk_to_host(_counted(mod, residual_to_device(x, mesh, x_dtype), start), mesh)

        return fn
    if step == "topk_shared":
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import topk_to_device
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import topk_to_host
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.topk_shared import TtSharedTopk

        mod = TtSharedTopk()

        def fn(ctx, x):
            # x (attn_norm) is the graph input only; the source full layer's top-k comes through ctx.extra["shared_topk"]
            # (swap_context; the model hands on the source indexer's device tensor instead)
            return topk_to_host(_counted(mod, topk_to_device(ctx.extra["shared_topk"], mesh)), mesh)

        return fn
    if step == "attention":
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import (
            kvpe_cache_load,
            kvpe_rows_to_device,
            topk_to_device,
        )
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

        mod = _attention(mesh, spec, layer)
        x_dtype = getattr(ttnn, settings.get("ATTN_NORM_DTYPE"))  # attn_norm's output dtype, as the device hands it on

        def fn(ctx, x, topk):
            start = ctx.extra.get("prefix_len", ctx.start)
            tk = topk_to_device(topk, mesh)
            if mod.owns_kv:
                prefix = ctx.extra.get("state_prefix", {})
                kvpe_cache_load(mod, prefix.get("kv_latent"), prefix.get("k_rope"), start, mesh)
                return residual_to_host(_counted(mod, residual_to_device(x, mesh, x_dtype), tk, start), mesh)
            # shared-KV layer: its source layer (run earlier in the chunk) has written rows [0, end) of the region the
            # module reads and hands on its bf16 rows of this chunk (swap_context: the golden's source-layer state)
            src = ctx.extra["shared_kv"]
            end = start + mod.chunk
            kvpe_cache_load(mod, src["kv_latent"], src["k_rope"], end, mesh)
            kvpe = kvpe_rows_to_device(mod, src["kv_latent"][start:end], src["k_rope"][start:end], mesh)
            out = residual_to_host(_counted(mod, residual_to_device(x, mesh, x_dtype), tk, start, 0, None, kvpe), mesh)
            ttnn.deallocate(kvpe)
            return out

        return fn
    if step in ("attn_residual", "mlp_residual", "moe_combine"):
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.residual import TtResidualAdd

        mod = TtResidualAdd()

        def fn(ctx, a, b):
            return residual_to_host(_counted(mod, residual_to_device(a, mesh), residual_to_device(b, mesh)), mesh)

        return fn
    if step in ("mlp", "shared_expert"):
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.mlp import TtMlp

        key = (id(mesh), layer, step)
        if key not in _MODULES:
            # the MoE layers' shared expert is the dense SwiGLU with its own weights (reference model's swiglu)
            p = "shared_" if step == "shared_expert" else ""
            w = _layer_weights(spec, layer, [p + "w1", p + "w3", p + "w2"])
            _MODULES[key] = TtMlp(mesh, w[p + "w1"], w[p + "w3"], w[p + "w2"])
        mod = _MODULES[key]

        def fn(ctx, x):
            return residual_to_host(_counted(mod, residual_to_device(x, mesh)), mesh)

        return fn
    if step == "router":
        from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import router_to_host
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.router import TtRouter

        key = (id(mesh), layer, step)
        if key not in _MODULES:
            cfg = Config(spec.hf_dir)
            _MODULES[key] = TtRouter(mesh, _layer_weights(spec, layer, ["gate"])["gate"], cfg.top_k, cfg.routed_scale)
        mod = _MODULES[key]

        def fn(ctx, x):
            return router_to_host(_counted(mod, residual_to_device(x, mesh)), mesh)

        return fn
    if step == "experts":
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import router_to_device

        mod = _experts(mesh, spec, layer)

        def fn(ctx, x, routing):
            return residual_to_host(_counted(mod, residual_to_device(x, mesh), router_to_device(routing, mesh)), mesh)

        return fn
    raise NotImplementedError(f"no device module for {step} yet")


def _new_module(mesh, spec, layer: int, step: str, **cache):
    """A new device module of one step (TtModel's ``build``)."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config

    if step in ("attn_norm", "ffn_norm"):
        return _norm(mesh, spec, layer, step)
    if step == "indexer":
        return _new_indexer(mesh, spec, layer, **cache)
    if step == "attention":
        return _new_attention(mesh, spec, layer, **cache)
    if step == "topk_shared":
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.topk_shared import TtSharedTopk

        return TtSharedTopk()
    if step in ("attn_residual", "mlp_residual", "moe_combine"):
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.residual import TtResidualAdd

        return TtResidualAdd()
    if step in ("mlp", "shared_expert"):
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.mlp import TtMlp

        p = "shared_" if step == "shared_expert" else ""
        w = _layer_weights(spec, layer, [p + "w1", p + "w3", p + "w2"])
        return TtMlp(mesh, w[p + "w1"], w[p + "w3"], w[p + "w2"])
    if step == "router":
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.router import TtRouter

        cfg = Config(spec.hf_dir)
        return TtRouter(mesh, _layer_weights(spec, layer, ["gate"])["gate"], cfg.top_k, cfg.routed_scale)
    if step == "experts":
        import torch

        from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, dequant, expert_store
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.experts import TtExperts, cache_complete
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS

        cfg = Config(spec.hf_dir)
        cache_dir = experts_cache_dir(spec, layer)
        weight = None
        if not cache_complete(cache_dir):
            le = expert_store(checkpoint(spec.hf_dir), cfg.n_experts).get(layer)
            weight = lambda e, m: dequant(le.w[(e, m)], le.s[(e, m)], torch.bfloat16)  # noqa: E731  (exact in bf16)
        return TtExperts(
            mesh,
            weight,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            rows_per_chip=_seq(spec)[1] // mesh.shape[SP_AXIS],
            cache_dir=cache_dir,
        )
    raise NotImplementedError(f"no device module for {step}")


def experts_cache_dir(spec, layer: int):
    """Layer ``layer``'s routed-expert weight cache under the spec's ``weights.cache`` (the runner's weight_cache_path,
    already per mesh), one per expert weight dtype; None without one."""
    from pathlib import Path

    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

    root = spec.get("weights.cache")
    if not root:
        return None
    return Path(root) / f"experts_{settings.get('EXPERTS_WEIGHT_DTYPE')}" / f"layer_{layer:02d}"


def device_model(mesh, spec, layers, lm_head=True, num_users=None, caches=None):
    """The all-device model (model.py TtModel): every step of every block a device module, the KV cache its state.
    Each layer's modules are built on its first call; its host weights are dropped once built. num_users / caches: the
    runner's slots and caches (tt/runners/adapters/mistral_large_4.py); None: the model's own (NUM_USERS)."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4._bringup_interface import weight_dtype
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config, Reference
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, load_dense
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.model import TtModel

    cfg = Config(spec.hf_dir)
    names = {"embed": "tok_embeddings.weight", "norm": "norm.weight"} | ({"head": "output.weight"} if lm_head else {})
    w = load_dense(checkpoint(spec.hf_dir), names, weight_dtype(spec))
    view = types.SimpleNamespace(cfg=cfg)  # the reference's block graph needs only its config, not its weights

    class _Model(TtModel):
        def layer(self, i, h, start, state, end=None):
            built = i in self.mods
            out = _counted(super().layer, i, h, start, state, end)
            if not built:
                _WEIGHTS.pop((str(spec.hf_dir), i), None)
            return out

    max_seq, chunk = _seq(spec)
    return _Model(
        mesh,
        cfg,
        layers,
        max_seq=max_seq,
        chunk=chunk,
        build=lambda layer, step, **cache: _new_module(mesh, spec, layer, step, **cache),
        embed_weight=w["embed"],
        norm_weight=w["norm"],
        lm_head_weight=w.get("head"),
        **({"num_users": num_users} if num_users is not None else {}),
        caches=caches,
        block_graph=lambda i: Reference.block_graph(view, i),
    )
