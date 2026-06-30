# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tt-symbiote-driven Cosmos3-Super-Image2Video pipeline factory.

Phase 1 MVP. Vendors the reference `Cosmos3OmniTransformer` and
`Cosmos3OmniPipeline` from `diffusers/main` into
`models/tt_dit/experimental/cosmos3_i2v/reference/` so we can stay on
tt-metal's pinned diffusers 0.35.1 / transformers 4.53.

Two factories:

  - `build_cosmos3_i2v_transformer_only(device)` — loads JUST the 64B
    transformer trunk from the HF model's `transformer/` subdir and
    wraps it with tt-symbiote. Use this for the smoke test (the full
    pipeline depends on Qwen3VLVisionModel which needs transformers
    >= 4.57; that bump is a separate decision).

  - `build_cosmos3_i2v_pipeline(device)` — loads the full vendored
    pipeline. Will fail in transformers 4.53 at the vision_encoder
    load step; left as the Phase 1 target once transformers is
    bumped or Qwen3VLVisionModel is also vendored.
"""

from __future__ import annotations

from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO

_MODEL_NAME = "cosmos3-i2v"
_TRANSFORMER_SUBFOLDER = "transformer"


def _maybe_load_linear_cache(
    *,
    all_modules,
    device,
    linear_cls,
    weight_dtype,
    subfolder: str,
    use_cache: bool,
):
    """Run sharded-Linear weights through `tt_dit.utils.cache.load_model`.

    On a cache hit, every replaced Linear's `tt_weight`/`tt_bias` is populated
    directly from disk via `ttnn.load_tensor`; the per-module
    `preprocess_weights()` + `move_weights_to_device()` pump is skipped (we
    flip the lifecycle flags so the base class short-circuits). On a miss,
    we still walk the modules and write the resulting on-device tensors back
    to the cache for next time.

    No-ops (returns `None`) when `use_cache=False` or `TT_DIT_CACHE_DIR` is
    unset — caller then falls back to the original preprocess/move flow.
    """
    if not use_cache:
        return None

    from models.tt_dit.experimental.cosmos3_i2v.tt_modules.cached_linear import (
        CosmosLinearBank,
        _dtype_key,
        build_parallel_config,
    )
    from models.tt_dit.utils import cache

    if not cache.cache_dir_is_set():
        return None

    mesh_shape = tuple(device.shape)
    bank = CosmosLinearBank(
        all_modules,
        device=device,
        weight_dtype=weight_dtype,
        linear_cls=linear_cls,
    )
    parallel_config = build_parallel_config(mesh_shape)
    dtype_key = _dtype_key(weight_dtype)

    cache.load_model(
        bank,
        model_name=_MODEL_NAME,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        dtype=dtype_key,
        get_torch_state_dict=bank.torch_state_dict,
    )

    bank.sync_to_tt_modules()
    return bank


def _move_uncached_modules(all_modules, *, skip_classes):
    """Run preprocess + move-to-device on every module that the cache layer didn't handle."""
    for tt_module in all_modules.values():
        if isinstance(tt_module, skip_classes):
            continue
        tt_module.preprocess_weights()
        tt_module.move_weights_to_device()


def _resolve_weight_dtype(weight_dtype, shard_linear: bool):
    """Pick the effective on-device weight dtype for caching purposes."""
    import ttnn

    if not shard_linear:
        return ttnn.bfloat16
    return weight_dtype if weight_dtype is not None else ttnn.bfloat16


def _pick_linear_class(
    *,
    shard_linear: bool,
    weight_dtype,
    sharded_base,
    sharded_factory,
    replicated,
    default_dtype,
):
    """Resolve the Linear replacement class for tt-symbiote's `nn_to_ttnn` dict.

    Centralizes the (shard_linear, weight_dtype) → class mapping so both
    builders pick consistently. When `shard_linear=False` the stock
    `TTNNLinearLLamaBFloat16` is used (it's bfloat16-only — `weight_dtype`
    is ignored and we warn-on-mismatch).
    """
    if not shard_linear:
        if weight_dtype is not None and weight_dtype != default_dtype:
            raise ValueError(
                f"weight_dtype={weight_dtype} ignored when shard_linear=False — "
                "the stock TTNNLinearLLamaBFloat16 is bfloat16-only. "
                "Set shard_linear=True to choose a different precision."
            )
        return replicated
    if weight_dtype is None or weight_dtype == default_dtype:
        return sharded_base
    return sharded_factory(weight_dtype)


_NUM_KEY_VALUE_HEADS = 8  # Cosmos3 config: matches Cosmos3OmniTransformer's num_key_value_heads


def _guard_native_joint_attn(*, native_joint_attn: bool | None, mesh_device, shard_linear: bool) -> bool:
    """Decide whether to enable the Phase 2 native joint-attention swap.

    The native `Cosmos3JointAttention` shards Q/K/V/out projections via
    `ColParallelLinear`/`RowParallelLinear` on the TP axis (picked as the
    larger axis of the mesh). Constraint: `tp_factor` must divide
    `num_key_value_heads` (8) — TP can't split a single KV head. So
    valid tp factors are {1, 2, 4, 8}. SP > 1 isn't exercised by the
    attention yet — the gen-pathway ring SDPA is the follow-up — so a
    multi-axis mesh like (2, 4) currently uses TP only along the larger
    axis (sp axis idle).

    Tri-state semantics:
      - `None` (auto): enable when `tp_factor` divides num_kv_heads (8);
        otherwise fall back to Phase 1 host attention silently.
      - `True`: enable; raise `ValueError` if the config can't support it.
      - `False`: explicit opt-out; always Phase 1.

    Returns the effective bool.
    """
    if native_joint_attn is False:
        return False

    mesh_shape = tuple(mesh_device.shape)
    tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
    tp_factor = mesh_shape[tp_axis]
    fits = _NUM_KEY_VALUE_HEADS % tp_factor == 0

    if native_joint_attn is True and not fits:
        msg = (
            f"native_joint_attn=True with tp_factor={tp_factor} but num_key_value_heads={_NUM_KEY_VALUE_HEADS} "
            f"is not divisible (mesh={mesh_shape}). Pick a mesh whose larger axis divides "
            f"{_NUM_KEY_VALUE_HEADS}, pass native_joint_attn=None to auto-fall-back, or "
            "native_joint_attn=False to stay on Phase 1 explicitly."
        )
        raise ValueError(msg)

    if not fits:
        print(
            f"[cosmos3-i2v] native_joint_attn=auto: falling back to Phase 1 host attention "
            f"(mesh={mesh_shape} → tp_factor={tp_factor} doesn't divide num_key_value_heads={_NUM_KEY_VALUE_HEADS}).",
            flush=True,
        )
        return False

    return True


def _build_replacement_dict(*, linear_cls, ttnn_silu_cls, native_joint_attn: bool):
    """Compose the tt-symbiote `nn_to_ttnn` replacement map.

    `native_joint_attn=True` swaps every `Cosmos3VLTextMoTDecoderLayer` for
    the Phase 2 native composite (RMSNorms + native joint-attention +
    native MLPs + residual adds, all on device). The recursive walk
    stops at the layer boundary, so the inner `nn.Linear`/`nn.SiLU` and
    `Cosmos3PackedMoTAttention`/`Cosmos3VLTextMLP` instances aren't
    visited individually for that layer — the native composite owns
    them. The per-attention and per-MLP swap entries stay registered
    as a fallback in case any stray instance exists outside the
    decoder-layer stack (none today, but cheap insurance).

    `nn.Linear`/`nn.SiLU` entries still apply to modules outside the
    decoder-layer stack: proj_in / proj_out / lm_head / embed_tokens /
    time_embedder.
    """
    from torch import nn

    nn_to_ttnn: dict = {
        nn.Linear: linear_cls,
        nn.SiLU: ttnn_silu_cls,
    }
    if native_joint_attn:
        from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
            Cosmos3PackedMoTAttention,
            Cosmos3VLTextMLP,
            Cosmos3VLTextMoTDecoderLayer,
        )
        from models.tt_dit.experimental.cosmos3_i2v.tt_modules.decoder_layer import TTNNCosmos3VLTextMoTDecoderLayer
        from models.tt_dit.experimental.cosmos3_i2v.tt_modules.joint_attention import TTNNCosmos3JointAttention
        from models.tt_dit.experimental.cosmos3_i2v.tt_modules.mlp import TTNNCosmos3VLTextMLP

        nn_to_ttnn[Cosmos3VLTextMoTDecoderLayer] = TTNNCosmos3VLTextMoTDecoderLayer
        nn_to_ttnn[Cosmos3PackedMoTAttention] = TTNNCosmos3JointAttention
        nn_to_ttnn[Cosmos3VLTextMLP] = TTNNCosmos3VLTextMLP
    return nn_to_ttnn


def build_cosmos3_i2v_transformer_only(
    device,
    *,
    dtype=None,
    hf_repo: str = HF_REPO,
    max_layers: int | None = None,
    shard_linear: bool = False,
    weight_dtype=None,
    use_cache: bool = True,
    native_joint_attn: bool | None = None,
):
    """Load only the Cosmos3OmniTransformer trunk and wrap with tt-symbiote.

    Returns the (wrapped) torch transformer module ready for forward calls.
    Caller is responsible for providing inputs matching the forward signature
    (see `reference/transformer_cosmos3.py` for the call surface).

    If `max_layers` is set, truncates the decoder layer stack to that many
    layers AFTER load. Useful when the full 64-layer model doesn't fit on
    the target mesh.

    If `shard_linear=True`, uses `TTNNLinearMeshShard` instead of the stock
    replicating `TTNNLinearLLamaBFloat16`. Each `nn.Linear` weight is then
    sharded along its output-feature dim across mesh axis 1. Required to
    fit the full 64B model on Galaxy; not useful on a 1x1 mesh.

    `weight_dtype` selects the on-device precision when `shard_linear=True`
    (no-op otherwise — the stock Linear is bfloat16-only). Defaults to
    `ttnn.bfloat16`. Set to `ttnn.bfloat8_b` for BFP8 — required to fit on
    WH LoudBox / T3K (12 GB per chip).

    `native_joint_attn` controls the Phase 2 swap of
    `Cosmos3PackedMoTAttention` for the native TT joint-attention block
    (projections, QK-norm, RoPE, both SDPA pathways, output projection
    all on device):
      - `None` (default, auto): enable on 1x1 mesh + `shard_linear=False`;
        silently fall back to Phase 1 otherwise.
      - `True`: enable; raise if config can't support it (>1 chip or
        sharded — the MVP doesn't yet do TP/SP).
      - `False`: explicit opt-out; always Phase 1 host attention.
    """
    import torch

    import ttnn
    from models.experimental.tt_symbiote.modules.activation import TTNNSilu
    from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLamaBFloat16
    from models.experimental.tt_symbiote.utils.device_management import set_device
    from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3OmniTransformer
    from models.tt_dit.experimental.cosmos3_i2v.tt_modules.linear_mesh_shard import (
        TTNNLinearMeshShard,
        make_sharded_linear_class,
    )

    if dtype is None:
        dtype = torch.bfloat16

    transformer = Cosmos3OmniTransformer.from_pretrained(hf_repo, subfolder="transformer", torch_dtype=dtype)

    if max_layers is not None and max_layers < len(transformer.layers):
        del transformer.layers[max_layers:]
        transformer.config.num_hidden_layers = max_layers

    # Count params BEFORE wrapping. After tt-symbiote replaces nn.Linear with
    # TTNNLinearLLamaBFloat16, the replaced modules no longer expose
    # `_parameters`, so `transformer.parameters()` crashes.
    param_count = sum(p.numel() for p in transformer.parameters())
    num_layers = len(transformer.layers)

    linear_cls = _pick_linear_class(
        shard_linear=shard_linear,
        weight_dtype=weight_dtype,
        sharded_base=TTNNLinearMeshShard,
        sharded_factory=make_sharded_linear_class,
        replicated=TTNNLinearLLamaBFloat16,
        default_dtype=ttnn.bfloat16,
    )
    native_joint_attn = _guard_native_joint_attn(
        native_joint_attn=native_joint_attn,
        mesh_device=device,
        shard_linear=shard_linear,
    )
    nn_to_ttnn = _build_replacement_dict(
        linear_cls=linear_cls,
        ttnn_silu_cls=TTNNSilu,
        native_joint_attn=native_joint_attn,
    )
    all_modules = register_module_replacement_dict(transformer, nn_to_ttnn, model_config=None)
    set_device(transformer, device)

    effective_weight_dtype = _resolve_weight_dtype(weight_dtype, shard_linear=shard_linear)
    bank = _maybe_load_linear_cache(
        all_modules=all_modules,
        device=device,
        linear_cls=linear_cls if shard_linear else TTNNLinearMeshShard,
        weight_dtype=effective_weight_dtype,
        subfolder=_TRANSFORMER_SUBFOLDER,
        # Cache only applies to the sharded path; replicating Linear has no
        # cross-chip layout to save and would dump 32× duplicated tensors.
        use_cache=use_cache and shard_linear,
    )
    # Replicated (non-sharded) Linears and modules the cache didn't touch
    # still need the preprocess + move-to-device pump.
    skip_classes = (linear_cls,) if bank is not None else ()
    _move_uncached_modules(all_modules, skip_classes=skip_classes)

    torch.set_grad_enabled(False)
    return transformer, all_modules, param_count, num_layers


def build_cosmos3_i2v_pipeline(
    device,
    *,
    dtype=None,
    hf_repo: str = HF_REPO,
    enable_vae_tiling: bool = True,
    shard_linear: bool = False,
    weight_dtype=None,
    use_cache: bool = True,
    native_joint_attn: bool | None = None,
):
    """Construct the full tt-symbiote-wrapped Cosmos3 I2V pipeline.

    Loads the vendored `Cosmos3OmniPipeline` from HF, which pulls in the
    transformer (64B), VAE (AutoencoderKLWan), vision encoder
    (Qwen3VLVisionModel), scheduler (UniPC), and Qwen2 tokenizer. Then
    tt-symbiote replaces `nn.Linear` (sharded if `shard_linear=True`) and
    `nn.SiLU` in the transformer and vision encoder.

    `weight_dtype` selects the on-device weight precision when
    `shard_linear=True` (see `build_cosmos3_i2v_transformer_only`).

    `native_joint_attn` is the same tri-state flag (None=auto, True=force,
    False=off) as in `build_cosmos3_i2v_transformer_only` and only affects
    the transformer trunk — the vision encoder uses its own attention class.

    Requires transformers >= 4.57 for `Qwen3VLVisionModel`.
    """
    import torch

    import ttnn
    from models.experimental.tt_symbiote.modules.activation import TTNNSilu
    from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLamaBFloat16
    from models.experimental.tt_symbiote.utils.device_management import set_device
    from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
    from models.tt_dit.experimental.cosmos3_i2v.reference.pipeline_cosmos3_omni import Cosmos3OmniPipeline
    from models.tt_dit.experimental.cosmos3_i2v.tt_modules.linear_mesh_shard import (
        TTNNLinearMeshShard,
        make_sharded_linear_class,
    )

    if dtype is None:
        dtype = torch.bfloat16

    # The HF model_index.json declares `"transformer": ["diffusers", "Cosmos3OmniTransformer"]`,
    # so `from_pretrained` does `getattr(diffusers, "Cosmos3OmniTransformer")`. That class
    # doesn't exist in diffusers 0.35.1 (it's on main). Inject our vendored version onto
    # the diffusers module so the lookup succeeds. Same for the pipeline itself in case any
    # subcomponent loader cross-references it.
    import diffusers as _diffusers

    from models.tt_dit.experimental.cosmos3_i2v.reference.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3OmniTransformer

    _diffusers.Cosmos3OmniTransformer = Cosmos3OmniTransformer
    _diffusers.Cosmos3OmniPipeline = Cosmos3OmniPipeline
    _diffusers.Cosmos3AVAEAudioTokenizer = Cosmos3AVAEAudioTokenizer

    linear_cls = _pick_linear_class(
        shard_linear=shard_linear,
        weight_dtype=weight_dtype,
        sharded_base=TTNNLinearMeshShard,
        sharded_factory=make_sharded_linear_class,
        replicated=TTNNLinearLLamaBFloat16,
        default_dtype=ttnn.bfloat16,
    )
    native_joint_attn = _guard_native_joint_attn(
        native_joint_attn=native_joint_attn,
        mesh_device=device,
        shard_linear=shard_linear,
    )
    transformer_nn_to_ttnn = _build_replacement_dict(
        linear_cls=linear_cls,
        ttnn_silu_cls=TTNNSilu,
        native_joint_attn=native_joint_attn,
    )
    # Vision encoder uses its own attention class; only the trunk gets the joint-attn swap.
    vision_nn_to_ttnn = _build_replacement_dict(
        linear_cls=linear_cls,
        ttnn_silu_cls=TTNNSilu,
        native_joint_attn=False,
    )

    # `enable_safety_checker=False` skips the CosmosSafetyChecker init, which would
    # otherwise require the `cosmos_guardrail` package (not installed in our venv).
    pipe = Cosmos3OmniPipeline.from_pretrained(hf_repo, torch_dtype=dtype, enable_safety_checker=False)
    if enable_vae_tiling and hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()

    # UniPC + use_karras_sigmas (the Cosmos3 default) produces a timesteps array
    # where two early values can round to the same integer. diffusers'
    # `index_for_timestep` picks `index_candidates[1]` (the second occurrence)
    # to support img2img mid-denoise starts, so `_step_index` starts at 1 instead
    # of 0 on a vanilla generate. After N steps it then walks one past the end,
    # making `len(timesteps) - step_index == 0` on the final iteration, which
    # zeroes `this_order` under `lower_order_final` and trips an assert inside
    # `scheduler.step()`. Setting `begin_index=0` makes `_init_step_index` use 0
    # directly and skip the duplicate-handling path.
    #
    # Subtlety: `scheduler.set_timesteps(...)` (called inside the pipeline's
    # `__call__`, every generate) explicitly resets `_begin_index = None`,
    # wiping out any value we set at build time. So a single
    # `set_begin_index(0)` here gets clobbered before the denoise loop runs.
    # Monkey-patch `set_timesteps` to re-apply `set_begin_index(0)` after
    # every call — covers both this build-time invocation (if any) and the
    # per-generate one inside `__call__`.
    if hasattr(pipe.scheduler, "set_begin_index"):
        _orig_set_timesteps = pipe.scheduler.set_timesteps

        def _set_timesteps_keep_begin_index_zero(*args, **kwargs):
            _orig_set_timesteps(*args, **kwargs)
            pipe.scheduler.set_begin_index(0)

        pipe.scheduler.set_timesteps = _set_timesteps_keep_begin_index_zero
        pipe.scheduler.set_begin_index(0)

    transformer_modules = register_module_replacement_dict(pipe.transformer, transformer_nn_to_ttnn, model_config=None)
    set_device(pipe.transformer, device)

    # vision_encoder is declared in model_index.json for other Cosmos3 runtimes
    # (NIM, vLLM-Omni) but not consumed by the diffusers pipeline (image
    # conditioning goes through the VAE). Wrap it only if `from_pretrained`
    # actually registered it on the pipe.
    vision_modules: dict = {}
    if hasattr(pipe, "vision_encoder") and pipe.vision_encoder is not None:
        vision_modules = register_module_replacement_dict(pipe.vision_encoder, vision_nn_to_ttnn, model_config=None)
        set_device(pipe.vision_encoder, device)

    effective_weight_dtype = _resolve_weight_dtype(weight_dtype, shard_linear=shard_linear)
    # Cache transformer and vision_encoder separately so their tensors don't
    # collide if model sizes drift between revisions.
    transformer_bank = _maybe_load_linear_cache(
        all_modules=transformer_modules,
        device=device,
        linear_cls=linear_cls if shard_linear else TTNNLinearMeshShard,
        weight_dtype=effective_weight_dtype,
        subfolder=_TRANSFORMER_SUBFOLDER,
        use_cache=use_cache and shard_linear,
    )
    vision_bank = _maybe_load_linear_cache(
        all_modules=vision_modules,
        device=device,
        linear_cls=linear_cls if shard_linear else TTNNLinearMeshShard,
        weight_dtype=effective_weight_dtype,
        subfolder="vision_encoder",
        use_cache=use_cache and shard_linear,
    )

    skip_classes = (linear_cls,) if (transformer_bank is not None or vision_bank is not None) else ()
    _move_uncached_modules(transformer_modules, skip_classes=skip_classes)
    _move_uncached_modules(vision_modules, skip_classes=skip_classes)

    torch.set_grad_enabled(False)
    return pipe
