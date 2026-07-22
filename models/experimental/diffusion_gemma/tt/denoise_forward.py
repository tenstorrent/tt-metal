# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN denoise-forward helpers for DiffusionGemma canvas attention.

Canvas queries attend to the frozen prompt prefix plus the current canvas as an
all-attend rectangle. The canonical path is maskless non-causal SDPA; explicit
masks are kept only for op A/B tests and non-canonical experiments. Sampling,
self-conditioning, and the multi-layer generation loop live in later W3/W4
helpers; this module owns the real denoise attention wiring.
"""

from __future__ import annotations

import os

import torch
import ttnn
from loguru import logger

from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    build_canvas_reveal_denoise_mask,
)
from models.experimental.diffusion_gemma.tt.diffusion_attention import denoise_attention
from models.experimental.diffusion_gemma.tt.expert_operations import (
    shared_mlp_forward,
    use_tanh_expert_activations,
)
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    _rms_norm_dram,
    build_self_conditioning,
    build_self_conditioning_embedding_weight,
    build_self_conditioning_embedding_weight_vocab_sharded,
)
from models.experimental.diffusion_gemma.weight_mapping import GEMMA4_LM_PREFIX, remap_state_dict

NEG = -1.0e9
TILE_SIZE = getattr(ttnn, "TILE_SIZE", 32)


def _terminal_sharded_enabled() -> bool:
    """Whether the TP-sharded denoise terminal is opted in (``DG_TERMINAL_SHARDED``, default off).

    When on, :func:`denoise_logits_forward` returns the per-device vocab shard (skipping the
    ~128 MiB/step full-vocab all-gather) and the terminal argmax/gumbel/entropy + self-cond
    soft-embedding run on the shard via the sharded ops in ``tt/sampling.py`` and
    ``tt/self_conditioning.py``. Default off keeps the full-vocab replicated path byte-identical
    (same gating discipline as ``DG_NORM_FULLCANVAS`` / ``DG_DEDUP_ARGMAX``)."""
    return os.environ.get("DG_TERMINAL_SHARDED", "0").lower() in ("1", "true", "yes", "on")


def default_self_conditioning_compute_kernel_config():
    """Select HiFi4/fp32 accumulation for the moderate-vocab full-softmax path.

    The production 262144-vocab path uses ordered online chunks in
    ``tt.self_conditioning`` and does not forward this matmul config.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def build_device_canvas_denoise_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    dtype=ttnn.bfloat16,
):
    """Build a `[1, 1, C, P+C]` denoise mask on device."""
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        layer_type=layer_type,
        sliding_window=sliding_window,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, prompt_len + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def reveal_mask_enabled() -> bool:
    """True when the fixed-max prefix + reveal-mask capture-once path is opted in.

    ``DG_DENOISE_REVEAL_MASK`` (default OFF). This is Phase-1 of the paged-prefix plan
    (``doc/optimize_perf/paged_prefix_denoise_design.md``): the denoise prefix K/V is read
    at a CONSTANT ``p_max`` span so the traced graph is shape-invariant across blocks
    (capture-once/replay-many), and the growing committed prefix is exposed purely through
    a persistent reveal mask that hides the uncommitted tail ``[prompt_len:p_max]``. Unlike
    ``DG_DENOISE_FROZEN_PREFIX`` it is multi-block CORRECT (later blocks re-read the mutated
    cache and attend all earlier-committed KV).
    """
    return os.environ.get("DG_DENOISE_REVEAL_MASK", "0").lower() in ("1", "true", "yes", "on")


def build_device_canvas_reveal_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    p_max: int,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    enforce_sliding_window: bool = False,
    dtype=ttnn.bfloat16,
):
    """Build a constant-shape ``[1, 1, C, p_max + C]`` reveal mask on device (Phase 1).

    Content reveals committed prefix ``[0:prompt_len]`` + all canvas, hides the uncommitted
    tail ``[prompt_len:p_max]`` with ``NEG``. ``enforce_sliding_window=False`` matches today's
    maskless all-attend production path (bit-exact to the recapture golden on the committed
    span); ``True`` additionally applies HF's bidirectional window (Phase 2, decision change).
    """
    mask = build_canvas_reveal_denoise_mask(
        prompt_len,
        canvas_len,
        p_max,
        layer_type=layer_type,
        sliding_window=sliding_window,
        enforce_sliding_window=enforce_sliding_window,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, p_max + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def _layer_type_for_denoise(tt_model, layer_idx: int) -> str | None:
    layer_types = getattr(getattr(tt_model, "hf_config", None), "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx]
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    return getattr(attn_config, "layer_type", None)


def _sliding_window_for_denoise(tt_model, layer_idx: int) -> int | None:
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    window = getattr(attn_config, "sliding_window", None)
    if window is not None:
        return window
    return getattr(getattr(tt_model, "hf_config", None), "sliding_window", None)


def _sliding_layer_needs_denoise_mask(prompt_len: int, canvas_len: int, sliding_window: int) -> bool:
    # HF's bidirectional sliding overlay allows abs(q_idx - kv_idx) <= sliding_window.
    return prompt_len + canvas_len - 1 > sliding_window


def _build_denoise_attn_mask_for_layer(
    tt_model,
    layer_idx: int,
    *,
    prompt_len: int,
    canvas_len: int,
    use_explicit_sliding_mask: bool = False,
    mask_builder=build_device_canvas_denoise_mask,
):
    if not use_explicit_sliding_mask:
        return None

    layer_type = _layer_type_for_denoise(tt_model, layer_idx)
    if layer_type != "sliding_attention":
        return None

    sliding_window = _sliding_window_for_denoise(tt_model, layer_idx)
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(f"sliding_attention layer {layer_idx} requires a positive sliding_window")
    if not _sliding_layer_needs_denoise_mask(prompt_len, canvas_len, sliding_window):
        return None

    return mask_builder(
        tt_model.mesh_device,
        prompt_len=prompt_len,
        canvas_len=canvas_len,
        layer_type="sliding_attention",
        sliding_window=sliding_window,
    )


def _deallocate_optional_tensor(tensor) -> None:
    if tensor is not None and hasattr(tensor, "deallocate"):
        tensor.deallocate(True)


def denoise_attention_forward(
    tt_model,
    *,
    layer_idx: int,
    prompt_hidden=None,
    prompt_kv=None,
    canvas_hidden,
    attn_mask=None,
    q_rope_offset: int | None = None,
):
    """Run one DiffusionGemma denoise attention layer on canvas hidden states.

    Args:
        tt_model: `Gemma4Model` carrying the reused DiffusionGemma decoder weights.
        layer_idx: decoder layer to run.
        prompt_hidden: frozen prompt hidden states `[1, 1, P, H]` on device.
        prompt_kv: optional frozen projected prompt `(K, V)` heads. This is the
            cache-shaped input used by the eventual paged encoder KV read path.
        canvas_hidden: current canvas hidden states `[1, 1, C, H]` on device.
        attn_mask: optional prebuilt `[1, 1, C, P+C]` additive mask on device.
            Leave unset for the canonical all-attend denoise path.

    Returns:
        The attention output for the canvas positions `[1, 1, C, H]`.
    """
    if (prompt_hidden is None) == (prompt_kv is None):
        raise ValueError("pass exactly one of prompt_hidden or prompt_kv")
    prompt_len = prompt_kv[0].shape[-2] if prompt_kv is not None else prompt_hidden.shape[-2]
    canvas_len = canvas_hidden.shape[-2]
    q_rope_offset = prompt_len if q_rope_offset is None else q_rope_offset
    kv_hidden = None if prompt_kv is not None else ttnn.concat([prompt_hidden, canvas_hidden], dim=2)
    out = denoise_attention(
        tt_model.layers[layer_idx].self_attn,
        canvas_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + canvas_len),
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        prefix_kv=prompt_kv,
        q_rope_offset=q_rope_offset,
    )
    if kv_hidden is not None:
        kv_hidden.deallocate(True)
    return out


def _prompt_source_len(prompt_source):
    return prompt_source[0].shape[-2] if isinstance(prompt_source, (tuple, list)) else prompt_source.shape[-2]


def _deallocate_prompt_source(prompt_source) -> None:
    if isinstance(prompt_source, (tuple, list)):
        for tensor in prompt_source:
            _deallocate_optional_tensor(tensor)
    else:
        _deallocate_optional_tensor(prompt_source)


# ---------------------------------------------------------------------------------------------
# HIGH-4 (dg-08 L1 pass): collapse the chunked RMSNorm to ONE full-canvas width-sharded rms_norm.
#
# DiffusionGemma chunks the 256-row canvas into 8x 32-row slices (``_chunked_norm_forward`` /
# ``_rms_norm_dram``) SO THAT each slice hits gemma4 RMSNorm's width-sharded fast path
# (``_forward_sharded``, block_h=1, 32-row-only); ``norm.forward`` on the full 256 rows falls to the
# slow plain-interleaved path (rms_norm.py:145-176). That costs 7 extra slices + 1 DRAM concat +
# 7 extra sharded-norm launches + 8 I2S/S2I round-trips PER norm call (~6-8 norm calls/layer x 30).
#
# RMSNorm normalizes each ROW independently over the hidden width, so block_h=8 (256 rows in one op)
# is per-row EQUIVALENT to 8x block_h=1 (the cross-core width reduction is per-row regardless of
# block_h). It is NOT, however, bit-identical: the bf16 reduction/accumulation ORDER differs between
# block_h=8 and 8x block_h=1, ~2e-6/norm (PCC 0.999998), which compounds over 30L x 48 steps under
# #48291 (no argmax cushion) and flips some committed tokens. So this runs one 256-row width-sharded
# rms_norm (reusing ``norm.tt_weight`` — reading the weight is data-use, NOT a gemma4 edit) and hands
# the L1 output straight back, dropping the slice/concat glue. MEASURED +15.8% @48 / +23.3% @12 traced
# (doc/optimize_perf/l1_residency.md). Gated DG_NORM_FULLCANVAS, default OFF (default path unchanged /
# bit-identical) until a decision-fidelity check vs HF clears the non-bit-identity for a default flip.
# ---------------------------------------------------------------------------------------------

_NORM_FULLCANVAS_CFG_CACHE = {}


def _norm_fullcanvas_enabled():
    return os.environ.get("DG_NORM_FULLCANVAS", "0") == "1"


def _build_fullcanvas_norm_cfg(mesh, seq_len, hidden_size):
    """Width-sharded rms_norm config for the FULL canvas (``seq_len`` rows, ``block_h=seq_len//32``).

    Largest core grid whose count divides the hidden tile-cols (mirrors gemma4
    RMSNorm._build_sharded_cfg, but block_h>1 for the whole canvas). Returns
    ``(input_memcfg, program_config)`` or ``None`` if no usable grid divides the width.
    """
    if hidden_size % TILE_SIZE != 0 or seq_len % TILE_SIZE != 0:
        return None
    key = (id(mesh), seq_len, hidden_size)
    cached = _NORM_FULLCANVAS_CFG_CACHE.get(key)
    if cached is not None:
        return cached
    tiles = hidden_size // TILE_SIZE
    grid = mesh.compute_with_storage_grid_size()
    best = None  # (num_cores, gx, gy)
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            n = gx * gy
            if tiles % n == 0 and (best is None or n > best[0]):
                best = (n, gx, gy)
    if best is None or best[0] == 1:
        _NORM_FULLCANVAS_CFG_CACHE[key] = None
        return None
    num_cores, gx, gy = best
    block_w = tiles // num_cores
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(seq_len, hidden_size // num_cores),
        core_grid=ttnn.CoreGrid(x=gx, y=gy),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w,
        block_h=seq_len // TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )
    cfg = (input_memcfg, program_config)
    _NORM_FULLCANVAS_CFG_CACHE[key] = cfg
    return cfg


def _fullcanvas_norm(norm, hidden_states):
    """One 256-row width-sharded rms_norm (HIGH-4). Returns normed [1,1,S,H] DRAM, or None if the
    width-sharded config is unavailable (caller falls back to the chunked path)."""
    cfg = _build_fullcanvas_norm_cfg(hidden_states.device(), hidden_states.shape[-2], hidden_states.shape[-1])
    if cfg is None:
        return None
    input_memcfg, program_config = cfg
    weight = getattr(norm, "tt_weight", None)
    x_sh = ttnn.to_memory_config(hidden_states, input_memcfg)
    out_sh = ttnn.rms_norm(
        x_sh,
        weight=weight,
        epsilon=norm.eps,
        program_config=program_config,
        memory_config=input_memcfg,
    )
    x_sh.deallocate(True)
    out = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)
    out_sh.deallocate(True)
    return out


def _chunked_norm_forward(norm, hidden_states, *, chunk_size: int = 32):
    if _norm_fullcanvas_enabled() and hidden_states.shape[-2] > chunk_size:
        out = _fullcanvas_norm(norm, hidden_states)
        if out is not None:
            return out
    if getattr(norm, "with_scale", True) is False and getattr(norm, "tt_weight", None) is None:
        return _rms_norm_dram(hidden_states, epsilon=norm.eps, chunk_size=chunk_size)
    seq_len = hidden_states.shape[-2]
    if seq_len <= chunk_size:
        return norm.forward(hidden_states)

    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = ttnn.slice(
            hidden_states,
            [0, 0, start, 0],
            [hidden_states.shape[0], hidden_states.shape[1], end, hidden_states.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunks.append(norm.forward(chunk))
        chunk.deallocate(True)
    out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in chunks:
        chunk.deallocate(True)
    return out


def _denoise_router_forward(router, hidden_states):
    normed = _chunked_norm_forward(router.norm, hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)

    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)

    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)

    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    top_k_values = ttnn.div(top_k_values, top_k_sum)
    top_k_sum.deallocate(True)

    dense_routing = ttnn.scatter(
        ttnn.zeros_like(router_probs),
        dim=-1,
        index=top_k_indices,
        src=top_k_values,
    )
    router_probs.deallocate(True)
    top_k_values.deallocate(True)
    top_k_indices.deallocate(True)

    if router.per_expert_scale is not None:
        dense_routing = ttnn.mul(dense_routing, router.per_expert_scale)

    return dense_routing


def _denoise_router_compact_forward(router, hidden_states):
    """Router forward that preserves compact top-k metadata for ragged MoE."""
    from models.experimental.diffusion_gemma.tt.sparse_moe import RaggedRouting

    normed = _chunked_norm_forward(router.norm, hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)
    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)
    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)
    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    router_probs.deallocate(True)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    normalized_values = ttnn.div(top_k_values, top_k_sum)
    top_k_values.deallocate(True)
    top_k_sum.deallocate(True)
    return RaggedRouting(normalized_values, top_k_indices, router.per_expert_scale)


def _denoise_moe_forward(moe, router_input, expert_input):
    # True-sparse token-gather MoE (~13x cheaper than the dense-128 path). Opt-in via env while
    # PCC / traced-t/s is validated; default flips once verified. See tt/sparse_moe.py.
    if os.environ.get("DG_SPARSE_MOE", "0") == "1":
        from models.experimental.diffusion_gemma.tt.sparse_moe import (
            compact_ragged_denoise_enabled,
            compact_ragged_denoise_forward,
            sparse_experts_forward,
        )

        if compact_ragged_denoise_enabled():
            routing = _denoise_router_compact_forward(moe.router, router_input)
            return compact_ragged_denoise_forward(moe.experts, expert_input, routing)
        dense_routing = _denoise_router_forward(moe.router, router_input)
        # Capacity must be zero-drop for diffusion correctness: real routing is highly
        # concentrated (measured max expert load 156-256 for a 256-token canvas), so the
        # old default of 32 silently discarded 41-84% of active routes per layer.
        capacity = int(os.environ.get("DG_SPARSE_MOE_CAPACITY", str(expert_input.shape[2])))
        out = sparse_experts_forward(moe.experts, expert_input, dense_routing, capacity=capacity)
        dense_routing.deallocate(True)
        return out
    dense_routing = _denoise_router_forward(moe.router, router_input)
    with use_tanh_expert_activations():
        return moe.experts(expert_input, dense_routing)


def _denoise_shared_mlp_forward(mlp, hidden_states):
    if os.environ.get("DG_GELU_TANH", "1") != "1":
        return mlp(hidden_states)
    return shared_mlp_forward(mlp, hidden_states)


def _denoise_layer_forward(
    tt_model, layer_idx, hidden_states, prompt_source, attn_mask, q_rope_offset, canvas_rope_provider=None
):
    layer = tt_model.layers[layer_idx]
    residual = hidden_states
    normed = _chunked_norm_forward(layer.input_layernorm, hidden_states)
    prefix_kv = prompt_source if isinstance(prompt_source, (tuple, list)) else None
    kv_hidden = None if prefix_kv is not None else ttnn.concat([prompt_source, normed], dim=2)
    # Cross-block-trace-reusable RoPE: when a canvas_rope_provider is supplied it returns a
    # CONSTANT-SHAPE [1,1,C,head_dim] buffer already holding cos/sin for the absolute canvas
    # positions [start_pos:start_pos+C] (updated per block OUTSIDE the trace); applying it at
    # start_offset=0 is bit-identical to slicing the growing [:, :, :start_pos+C, :] cache at
    # start_offset=start_pos (RoPE cos/sin depend only on absolute position). This keeps the
    # captured trace's RoPE tensor addresses/shapes fixed across blocks. Only valid on the
    # prefix_kv path (Q and canvas-K are both C rows); the kv_hidden recompute path (K spans the
    # full prompt+canvas) must keep the growing slice. See #47465.
    if canvas_rope_provider is not None:
        if prefix_kv is None:
            raise ValueError("canvas_rope_provider requires the prefix_kv denoise path")
        rope_mats = canvas_rope_provider(layer_idx)
        rope_offset = 0
    else:
        rope_mats = tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + hidden_states.shape[-2])
        rope_offset = q_rope_offset
    attn_output = denoise_attention(
        layer.self_attn,
        normed,
        rope_mats=rope_mats,
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        prefix_kv=prefix_kv,
        q_rope_offset=rope_offset,
    )
    if kv_hidden is not None:
        kv_hidden.deallocate(True)

    attn_output = _chunked_norm_forward(layer.post_attention_layernorm, attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = _chunked_norm_forward(layer.pre_feedforward_layernorm, hidden_states)
    mlp_output = _denoise_shared_mlp_forward(layer.shared_mlp, normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_1, mlp_output)
        mlp_output.deallocate(True)
        expert_input = _chunked_norm_forward(layer.pre_feedforward_layernorm_2, residual)
        expert_output = _denoise_moe_forward(layer.moe, residual, expert_input)
        expert_input.deallocate(True)
        expert_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_2, expert_output)
        expert_output.deallocate(True)
        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = _chunked_norm_forward(layer.post_feedforward_layernorm, hidden_states)
    combined = ttnn.add(residual, hidden_states)
    residual.deallocate(True)
    hidden_states.deallocate(True)
    if layer.layer_scalar != 1.0:
        scaled = ttnn.mul(combined, layer.layer_scalar)
        combined.deallocate(True)
        combined = scaled
    return combined


def denoise_hidden_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
    q_rope_offset: int | None = None,
    prompt_len: int | None = None,
    use_explicit_sliding_mask: bool = False,
    mask_builder=build_device_canvas_denoise_mask,
    canvas_rope_provider=None,
    reveal_mask_provider=None,
):
    """Run the DiffusionGemma denoise backbone to final hidden states.

    ``prompt_hidden_by_layer`` provides the frozen encoder-side attention source
    for each decoder layer. Entries can be either `[1, 1, P, H]` hidden tensors
    (legacy shim), projected `(K, V)` prompt heads, or a callable that lazily
    returns a per-layer prompt source. The production path is maskless all-attend;
    set ``use_explicit_sliding_mask`` only for HF-geometry A/B tests.
    """
    prompt_source_fn = prompt_hidden_by_layer if callable(prompt_hidden_by_layer) else None
    if prompt_source_fn is None and len(prompt_hidden_by_layer) != len(tt_model.layers):
        raise ValueError(
            f"prompt_hidden_by_layer has {len(prompt_hidden_by_layer)} entries but model has {len(tt_model.layers)} layers"
        )
    if prompt_len is None:
        if prompt_source_fn is not None:
            raise ValueError("prompt_len is required when prompt_hidden_by_layer is callable")
        prompt_len = _prompt_source_len(prompt_hidden_by_layer[0])

    hidden_states = canvas_hidden
    q_rope_offset = prompt_len if q_rope_offset is None else q_rope_offset
    for layer_idx in range(len(tt_model.layers)):
        if reveal_mask_provider is not None:
            # Paged-prefix Phase 1: persistent per-block reveal mask (hides the uncommitted
            # tail of the fixed p_max prefix read). Owned by the adapter — NOT deallocated here.
            attn_mask = reveal_mask_provider(layer_idx)
        else:
            attn_mask = _build_denoise_attn_mask_for_layer(
                tt_model,
                layer_idx,
                prompt_len=prompt_len,
                canvas_len=hidden_states.shape[-2],
                use_explicit_sliding_mask=use_explicit_sliding_mask,
                mask_builder=mask_builder,
            )
        prompt_source = (
            prompt_source_fn(layer_idx) if prompt_source_fn is not None else prompt_hidden_by_layer[layer_idx]
        )
        try:
            hidden_states = _denoise_layer_forward(
                tt_model,
                layer_idx,
                hidden_states,
                prompt_source,
                attn_mask,
                q_rope_offset,
                canvas_rope_provider=canvas_rope_provider,
            )
        finally:
            if prompt_source_fn is not None:
                _deallocate_prompt_source(prompt_source)
            if reveal_mask_provider is None:
                _deallocate_optional_tensor(attn_mask)
    final_hidden = _chunked_norm_forward(tt_model.norm, hidden_states)
    hidden_states.deallocate(True)
    return final_hidden


def denoise_logits_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
    q_rope_offset: int | None = None,
    prompt_len: int | None = None,
    use_explicit_sliding_mask: bool = False,
    canvas_rope_provider=None,
    reveal_mask_provider=None,
    return_sharded: bool = False,
):
    """Run a short-prompt DiffusionGemma denoise logits forward.

    The returned logits cover all canvas positions, which the diffusion sampler
    consumes each denoise step. ``return_sharded`` (default ``False``) forwards to
    ``_apply_lm_head`` so a TP-sharded terminal (``DG_TERMINAL_SHARDED``) can skip the
    per-step full-vocab all-gather; default off returns the full replicated logits
    exactly as before.
    """
    hidden_states = denoise_hidden_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
        q_rope_offset=q_rope_offset,
        prompt_len=prompt_len,
        use_explicit_sliding_mask=use_explicit_sliding_mask,
        canvas_rope_provider=canvas_rope_provider,
        reveal_mask_provider=reveal_mask_provider,
    )
    return tt_model._apply_lm_head(hidden_states, is_decode=False, return_sharded=return_sharded)


def denoise_terminal_reductions_sharded(
    logits_shard,
    *,
    temperature: float,
    offsets,
    mesh_config,
    ccl_manager,
    gumbel_noise_shard=None,
    dedup_argmax: bool | None = None,
):
    """Sharded terminal vocab reductions (E2 routing): ``(sampled, argmax, entropy)`` from the
    per-device vocab shard, mirroring ``denoise_loop._sample_and_argmax`` + ``token_entropy`` but on
    the shard via the ``tt/sampling.py`` sharded ops.

    ``argmax`` (committed token) is BIT-IDENTICAL to the replicated path; ``sampled`` matches the
    replicated ``gumbel_max`` (same tie rule; ``gumbel_noise_shard=None`` => temperature-scaled
    argmax, with the dedup fast path when ``DG_DEDUP_ARGMAX`` is on); ``entropy`` is the fp32
    distributed logsumexp (decision-gated, NOT bf16-bit-identical). The exact global max is computed
    ONCE and shared by argmax (implicitly, via the value compare) and entropy.

    This composes the sharded ops into the same three outputs ``denoise_step`` consumes; the
    downstream accept-mask / renoise / commit are vocab-agnostic and unchanged. Callers preallocate
    ``offsets`` (and any noise shard) OUTSIDE trace capture; see
    :meth:`DenoiseLogitsAdapter.prepare_sharded_terminal`.
    """
    from models.experimental.diffusion_gemma.tt import sampling as TS
    from models.experimental.diffusion_gemma.tt.denoise_loop import dedup_argmax_enabled

    if dedup_argmax is None:
        dedup_argmax = dedup_argmax_enabled()
    if dedup_argmax and gumbel_noise_shard is None and temperature > 0:
        argmax = TS.argmax_last_dim_sharded(logits_shard, offsets, mesh_config=mesh_config, ccl_manager=ccl_manager)
        sampled = ttnn.clone(argmax, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        sampled = TS.gumbel_max_sharded(
            logits_shard, temperature, gumbel_noise_shard, offsets, mesh_config=mesh_config, ccl_manager=ccl_manager
        )
        argmax = TS.argmax_last_dim_sharded(logits_shard, offsets, mesh_config=mesh_config, ccl_manager=ccl_manager)
    entropy = TS.token_entropy_sharded(
        logits_shard, temperature=temperature, mesh_config=mesh_config, ccl_manager=ccl_manager
    )
    return sampled, argmax, entropy


def collect_prompt_hidden_by_layer(tt_model, prompt_hidden):
    """Collect per-layer frozen prompt attention inputs for denoise K/V source.

    The DiffusionGemma decoder reads encoder K/V. Until the real paged encoder
    cache is threaded into this wrapper, this helper captures the tensor that
    feeds each layer's K/V projections: the prompt hidden states after that
    layer's input RMSNorm, while advancing the prompt through the normal causal
    layer stack. Returned tensors are owned by the caller.
    """
    hidden_states = prompt_hidden
    prompt_hidden_by_layer = []
    for layer_idx, layer in enumerate(tt_model.layers):
        prompt_hidden_by_layer.append(layer.input_layernorm.forward(hidden_states))
        hidden_states = layer(
            hidden_states,
            rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=hidden_states.shape[-2]),
            position_idx=None,
            page_table=None,
            kv_cache=None,
            is_decode=False,
        )
    hidden_states.deallocate(True)
    return prompt_hidden_by_layer


def collect_prompt_kv_by_layer(tt_model, prompt_hidden):
    """Collect per-layer frozen prompt K/V heads for denoise prefix attention.

    This uses the existing Gemma4 prefill ``keep_kv`` path to capture K/V after
    per-head norm and RoPE. Those tensors match the shape carried by KV caches,
    so this is the narrow interface the future paged encoder-cache read should
    populate.
    """
    hidden_states = prompt_hidden
    prompt_kv_by_layer = []
    for layer_idx, layer in enumerate(tt_model.layers):
        hidden_states = layer(
            hidden_states,
            rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=hidden_states.shape[-2]),
            position_idx=None,
            page_table=None,
            kv_cache=None,
            is_decode=False,
            keep_kv=True,
        )
        prompt_kv_by_layer.append(layer.self_attn._last_kv)
    hidden_states.deallocate(True)
    return prompt_kv_by_layer


def read_prompt_kv_cache_slice(kv_cache, *, prompt_len: int, seq_len_start: int = 0):
    """Read a frozen prompt K/V prefix from a contiguous Gemma4 KV cache.

    This is the non-paged cache adapter for W2: it reads the encoder-written K/V
    heads from `[B, heads, max_seq, head_dim]` cache tensors and returns the
    cache-shaped `(K, V)` prefix accepted by denoise attention. The underlying
    TTNN slice op requires full tiles along sequence, so bounds must be 32-aligned.
    """
    seq_len_end = seq_len_start + prompt_len
    if seq_len_start % ttnn.TILE_SIZE != 0 or seq_len_end % ttnn.TILE_SIZE != 0:
        raise ValueError("KV cache slice bounds must be multiples of 32")
    k_cache, v_cache = kv_cache
    # A slice spanning the ENTIRE cache seq dim (the reveal-mask fixed p_max == cache_len read)
    # ALIASES the cache buffer, and denoise_attention deallocates the returned prefix K/V — which
    # would free the model-owned cache ("Input Tensor is not allocated" on the next block). For the
    # full-span read, clone the cache DIRECTLY into a caller-owned copy (never slice/deallocate the
    # alias). Partial slices already produce an independent copy, so they take the cheap path.
    if seq_len_start == 0 and seq_len_end == int(k_cache.shape[2]):
        return (ttnn.clone(k_cache), ttnn.clone(v_cache))
    starts = [0, 0, seq_len_start, 0]
    k_ends = [k_cache.shape[0], k_cache.shape[1], seq_len_end, k_cache.shape[3]]
    v_ends = [v_cache.shape[0], v_cache.shape[1], seq_len_end, v_cache.shape[3]]
    return (
        ttnn.slice(k_cache, starts, k_ends, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.slice(v_cache, starts, v_ends, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )


def read_prompt_kv_cache_by_layer(
    tt_model,
    *,
    prompt_len: int,
    seq_len_start: int = 0,
    layer_idx: int | None = None,
    read_fn=read_prompt_kv_cache_slice,
):
    """Read frozen prompt K/V prefixes from every layer's Gemma4 KV cache.

    This is the production-shaped prompt source for the denoise adapter: one
    `(K, V)` tuple per decoder layer, rather than the early single-layer test
    shim. The returned tensors are owned by the caller.
    """
    if len(tt_model.tt_kv_cache) != len(tt_model.layers):
        raise ValueError(
            f"tt_kv_cache has {len(tt_model.tt_kv_cache)} layers but model has {len(tt_model.layers)} layers"
        )
    if layer_idx is not None:
        return read_fn(tt_model.tt_kv_cache[layer_idx], prompt_len=prompt_len, seq_len_start=seq_len_start)
    return [read_fn(kv_cache, prompt_len=prompt_len, seq_len_start=seq_len_start) for kv_cache in tt_model.tt_kv_cache]


class MutablePrefixKVReader:
    """Lazy per-layer contiguous-cache reader with a commit-advanced prefix span."""

    def __init__(self, tt_model, *, prompt_len: int, seq_len_start: int = 0, read_fn=read_prompt_kv_cache_by_layer):
        self.tt_model = tt_model
        self.prompt_len = int(prompt_len)
        self.seq_len_start = int(seq_len_start)
        self.read_fn = read_fn
        # Paged-prefix Phase 1 (reveal-mask): a CONSTANT read span decouples the traced
        # slice shape from the growing committed ``prompt_len``. When set, ``__call__`` always
        # reads ``read_span`` rows (so the trace never invalidates on prefix growth), and the
        # committed ``prompt_len`` only drives the reveal mask content + canvas RoPE anchor.
        self.read_span = None

    def set_read_span(self, p_max: int) -> None:
        p_max = int(p_max)
        if p_max % ttnn.TILE_SIZE != 0:
            raise ValueError(f"reveal-mask read span must be tile aligned, got {p_max}")
        if p_max < self.prompt_len:
            raise ValueError(f"read span {p_max} < committed prompt_len {self.prompt_len}")
        self.read_span = p_max

    def __call__(self, layer_idx: int):
        n = self.read_span if self.read_span is not None else self.prompt_len
        return self.read_fn(
            self.tt_model,
            prompt_len=n,
            seq_len_start=self.seq_len_start,
            layer_idx=layer_idx,
        )

    def set_prompt_len(self, prompt_len: int) -> None:
        prompt_len = int(prompt_len)
        if prompt_len < self.prompt_len:
            raise ValueError(f"frozen prefix cannot shrink: {self.prompt_len} -> {prompt_len}")
        if prompt_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"frozen prefix length must be tile aligned, got {prompt_len}")
        if self.read_span is not None and prompt_len > self.read_span:
            raise ValueError(f"committed prompt_len {prompt_len} exceeds reveal read span {self.read_span}")
        self.prompt_len = prompt_len

    def reset_prompt_len(self, prompt_len: int) -> None:
        """Reset the committed prefix at a request boundary on a fixed-span reader.

        Unlike :meth:`set_prompt_len`, this permits shrinking because a new request has
        overwritten the model-owned cache head. It is intentionally unavailable without a
        reveal-mask ``read_span``: shrinking a shape-baked prefix would replay stale prompt KV.
        """
        prompt_len = int(prompt_len)
        if self.read_span is None:
            raise RuntimeError("request-boundary prefix reset requires a fixed reveal-mask read span")
        if prompt_len < 0:
            raise ValueError(f"frozen prefix length must be non-negative, got {prompt_len}")
        if prompt_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"frozen prefix length must be tile aligned, got {prompt_len}")
        if prompt_len > self.read_span:
            raise ValueError(f"committed prompt_len {prompt_len} exceeds reveal read span {self.read_span}")
        self.prompt_len = prompt_len


def embed_canvas_tokens(tt_model, canvas_tokens):
    """Embed device canvas token ids into `[1, 1, C, H]` TILE hidden states."""
    if canvas_tokens.shape[0] != 1:
        raise ValueError("embed_canvas_tokens currently supports batch=1")
    if len(canvas_tokens.shape) == 4 and canvas_tokens.shape[-1] == 1:
        canvas_len = canvas_tokens.shape[-2]
        token_ids = ttnn.reshape(canvas_tokens, (canvas_tokens.shape[0], canvas_len))
        token_ids = ttnn.to_layout(token_ids, ttnn.ROW_MAJOR_LAYOUT)
    else:
        canvas_len = canvas_tokens.shape[-1]
        token_ids = canvas_tokens
    canvas_hidden = tt_model.embed_tokens(token_ids)
    if token_ids is not canvas_tokens:
        token_ids.deallocate(True)
    if len(canvas_hidden.shape) == 3:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    elif canvas_hidden.shape[-2] != canvas_len:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    return ttnn.to_layout(canvas_hidden, ttnn.TILE_LAYOUT)


def denoise_logits_from_tokens(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_tokens,
    self_conditioning=None,
    prev_logits=None,
    q_rope_offset: int | None = None,
    prompt_len: int | None = None,
    use_explicit_sliding_mask: bool = False,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
    self_conditioning_temperature: float = 1.0,
):
    """Embed canvas token ids, optionally self-condition, then run denoise logits."""
    canvas_hidden = embed_canvas_tokens(tt_model, canvas_tokens)
    if self_conditioning is not None:
        conditioned = self_conditioning.condition(
            canvas_hidden,
            prev_logits,
            self_conditioning_embedding_weight,
            compute_kernel_config=self_conditioning_compute_kernel_config,
            temperature=self_conditioning_temperature,
        )
        canvas_hidden.deallocate(True)
        canvas_hidden = conditioned
    return denoise_logits_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
        q_rope_offset=q_rope_offset,
        prompt_len=prompt_len,
        use_explicit_sliding_mask=use_explicit_sliding_mask,
    )


class DenoiseLogitsAdapter:
    """Stateful W2 logits callback for the W3 denoise controller.

    The controller calls ``logits_fn(canvas_tokens, step)``. This adapter turns
    that narrow callback into the real W2 path: token embedding, optional
    self-conditioning from the previous step's logits, and denoise logits forward.

    **Trace-safe self-conditioning** (``prepare_trace_safe_self_conditioning``).
    The default eager path threads the self-cond signal across steps as the
    previous step's *full* ``[1,1,C,vocab]`` logits, freshly allocated every step
    (``self.prev_logits``). A fixed Metal trace bakes in buffer addresses and one
    fixed unrolled graph, and this chained fresh-alloc cross-step state does not
    replay bit-exactly — the whole-loop trace committed argmax diverged (60.5%
    match) with self-cond ON while a self-cond-OFF / stateless loop traced at
    100% (``doc/optimize_perf/probe_traced_denoise_loop.py``). The trace-safe
    variant carries the cross-step state as the small ``[1,1,C,hidden]``
    soft-embedding **signal** in a persistent, preallocated buffer updated
    **in-place** each step (``ttnn.copy``), so its device address is fixed. Step 0
    still uses the ``condition(None)`` branch (``post_norm(embed)``) and only
    *writes* the buffer; steps 1+ *read* the buffer written by the immediately
    preceding step. This is bit-exact to the eager path (same ``soft_embedding``
    math, just computed at the producer step's end and copied) and carries no
    stale cross-block state (step 0 never reads the buffer). See #47465.
    """

    def __init__(
        self,
        tt_model,
        *,
        prompt_hidden_by_layer,
        self_conditioning=None,
        self_conditioning_embedding_weight=None,
        self_conditioning_compute_kernel_config=None,
        q_rope_offset: int | None = None,
        prompt_len: int | None = None,
        max_denoise_steps: int | None = None,
        temperature_start: float = 0.8,
        temperature_end: float = 0.4,
        logits_from_tokens=denoise_logits_from_tokens,
    ):
        self.tt_model = tt_model
        self.prompt_hidden_by_layer = prompt_hidden_by_layer
        self.prompt_len = prompt_len
        disable_self_conditioning = os.environ.get("DG_DISABLE_SELF_CONDITIONING", "0") == "1"
        self.self_conditioning = None if disable_self_conditioning else self_conditioning
        self.self_conditioning_embedding_weight = (
            None if disable_self_conditioning else self_conditioning_embedding_weight
        )
        self.self_conditioning_compute_kernel_config = self_conditioning_compute_kernel_config
        self.q_rope_offset = q_rope_offset
        self.logits_from_tokens = logits_from_tokens
        self.max_denoise_steps = max_denoise_steps
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.prev_logits = None
        # Trace-safe self-conditioning: persistent [1,1,C,hidden] signal buffer(s)
        # (for the single-step traced loop; KV-cache-style cross-replay state).
        self.trace_safe_self_conditioning = False
        self.signal_buf = None
        # Ping-pong (double-buffered) signal: read buffer != write buffer within a step.
        # Added to test whether an in-place signal_buf read+write was the "self-cond trace
        # race" — it is NOT: the in-place default is decision-fidelity-preserving and
        # ping-pong is BIT-IDENTICAL to it on device (the race was a probe harness bug —
        # a reused init buffer allocated after trace capture, clobbered by trace scratch;
        # see perf_progress.md session 8). Kept opt-in (default off) as a verified-equivalent
        # option; the shipped traced path uses the in-place default. See #47465.
        self.signal_ping_pong = False
        self.signal_buf_b = None
        # Cross-block-trace-reusable canvas RoPE (constant-shape per-layer-type buffers).
        self.use_canvas_rope = False
        self._canvas_rope_bufs = {}
        # Paged-prefix Phase 1 reveal mask (constant-shape [1,1,C,p_max+C] written input,
        # content refreshed per block OUTSIDE capture to reveal committed prefix / hide tail).
        self.use_reveal_mask = False
        self._reveal_mask_buf = None
        self._reveal_p_max = None
        self._reveal_canvas_len = None
        self._reveal_enforce_window = False
        # TP-sharded denoise terminal (DG_TERMINAL_SHARDED). Persistent constants allocated OUTSIDE
        # any trace by ``prepare_sharded_terminal``: per-device vocab offset + vocab-row-sharded
        # tied embedding table. Default off keeps the full-vocab replicated terminal.
        self.sharded_terminal = False
        self._vocab_offsets = None
        self._embedding_weight_sharded = None
        self._sharded_mesh_config = None
        self._sharded_ccl_manager = None

    def prepare_trace_safe_self_conditioning(self, *, canvas_len: int, dtype=ttnn.bfloat16, ping_pong: bool = False):
        """Preallocate the persistent in-place self-cond signal buffer OUTSIDE any trace.

        Intended for the **single-step traced loop** (``tt/denoise_loop.py``): one
        denoise step is captured as a Metal trace and replayed once per step, with
        the self-cond signal carried across replays in this persistent buffer,
        updated in-place each step — exactly the KV-cache pattern that a traced
        decode uses (and which, unlike a *whole-loop* trace with cross-step
        feedback, does not race: a single self-cond step traces at 100%, verified in
        ``probe_traced_denoise_loop.py`` STEPS=1). Uniform-graph: **every** step runs
        ``forward(embed, signal_buf)``; step 0 reads the zeroed buffer, which is
        bit-exact to the ``condition(None)`` (=``post_norm(embed)``) branch because
        ``forward`` with a zero signal has ``pre_norm(0)=0`` → gate/up/down all zero →
        ``post_norm(embed + 0)``. ``reset_signal_buffer`` must be called before each
        block's step 0 to re-zero. See #47465.
        """
        self.signal_ping_pong = ping_pong
        if self.self_conditioning is None:
            self.trace_safe_self_conditioning = True
            self.signal_buf = None
            self.signal_buf_b = None
            return
        hidden_size = self.self_conditioning.hidden_size
        if self.signal_buf is not None:
            self.signal_buf.deallocate(True)
        if self.signal_buf_b is not None:
            self.signal_buf_b.deallocate(True)
            self.signal_buf_b = None

        def _zeros():
            return ttnn.zeros(
                [1, 1, canvas_len, hidden_size],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.tt_model.mesh_device,
            )

        self.signal_buf = _zeros()
        if ping_pong:
            self.signal_buf_b = _zeros()
        self.trace_safe_self_conditioning = True

    def reset_signal_buffer(self):
        """Zero the persistent signal buffer (call before each block's step 0).

        Outside any trace (a fill is a WRITE, forbidden in capture). Step 0 then
        reads zeros -> ``forward(embed, 0) == post_norm(embed)``, matching the eager
        step-0 ``condition(None)`` branch bit-for-bit.
        """
        if self.signal_buf is not None:
            ttnn.mul(self.signal_buf, 0.0, output_tensor=self.signal_buf)
        if self.signal_buf_b is not None:
            ttnn.mul(self.signal_buf_b, 0.0, output_tensor=self.signal_buf_b)

    def _signal_read_write_bufs(self, step: int):
        """Return (read_buf, write_buf) for this step's self-cond signal.

        In-place (default): read == write == signal_buf. Ping-pong: even steps read
        buffer A and write buffer B, odd steps swap — so step N reads exactly the
        buffer step N-1 wrote, and no step reads+writes the same address in-place.
        """
        if not self.signal_ping_pong:
            return self.signal_buf, self.signal_buf
        if step % 2 == 0:
            return self.signal_buf, self.signal_buf_b
        return self.signal_buf_b, self.signal_buf

    # --- Cross-block-trace-reusable canvas RoPE (second single-step-trace blocker) ---
    #
    # The single-step trace also bakes in the RoPE cos/sin tensors used inside denoise
    # attention. The eager path passes ``_get_rope_mats(seq_len=start_pos+C)`` (a GROWING
    # slice) and applies it at ``start_offset=start_pos`` — so a trace captured on block N
    # is only valid for block N's start_pos. These methods preallocate a CONSTANT-SHAPE
    # ``[1,1,C,head_dim]`` canvas RoPE buffer per layer-type whose CONTENT (the cos/sin for
    # absolute positions ``[start_pos:start_pos+C]``) is refreshed per block OUTSIDE the trace;
    # ``_denoise_layer_forward`` then applies it at ``start_offset=0``. Because RoPE cos/sin
    # depend only on absolute position, that is bit-identical to the growing-slice path, and
    # the trace's RoPE tensor addresses/shapes stay fixed across blocks. See #47465.

    def prepare_canvas_rope_buffers(self, *, canvas_len: int):
        """Preallocate per-layer-type constant-shape canvas RoPE buffers OUTSIDE any trace.

        The buffers are cloned from the FIRST-``canvas_len`` slice of each layer-type's real
        RoPE cache, so their dtype / layout / memory_config match the cache exactly (a later
        ``ttnn.copy`` into them from an offset slice is then a same-spec device copy). Content
        is overwritten per block by ``update_canvas_rope_buffers``.
        """
        tt_model = self.tt_model
        self._canvas_rope_len = canvas_len
        self._canvas_rope_bufs = {}
        layer_types = tt_model.hf_config.layer_types
        for layer_idx in range(len(tt_model.layers)):
            layer_type = layer_types[layer_idx]
            if layer_type in self._canvas_rope_bufs:
                continue
            cos_full, sin_full = tt_model._get_rope_mats(layer_idx)
            bufs = []
            for full in (cos_full, sin_full):
                head = ttnn.slice(full, [0, 0, 0, 0], [full.shape[0], full.shape[1], canvas_len, full.shape[3]])
                bufs.append(ttnn.clone(head))
                head.deallocate(True)
            self._canvas_rope_bufs[layer_type] = (bufs[0], bufs[1])
        self.use_canvas_rope = True

    def update_canvas_rope_buffers(self, start_pos: int):
        """Refresh canvas RoPE buffer content for a block's absolute ``start_pos`` (OUTSIDE trace)."""
        if not getattr(self, "use_canvas_rope", False):
            return
        if start_pos % 32 != 0:
            raise ValueError(f"canvas RoPE start_pos must be a 32-tile multiple, got {start_pos}")
        tt_model = self.tt_model
        C = self._canvas_rope_len
        layer_types = tt_model.hf_config.layer_types
        done = set()
        for layer_idx in range(len(tt_model.layers)):
            layer_type = layer_types[layer_idx]
            if layer_type in done:
                continue
            cos_full, sin_full = tt_model._get_rope_mats(layer_idx)
            cos_buf, sin_buf = self._canvas_rope_bufs[layer_type]
            for full, buf in ((cos_full, cos_buf), (sin_full, sin_buf)):
                sliced = ttnn.slice(
                    full,
                    [0, 0, start_pos, 0],
                    [full.shape[0], full.shape[1], start_pos + C, full.shape[3]],
                )
                ttnn.copy(sliced, buf)
                sliced.deallocate(True)
            done.add(layer_type)

    def _canvas_rope_provider(self, layer_idx):
        layer_type = self.tt_model.hf_config.layer_types[layer_idx]
        return self._canvas_rope_bufs[layer_type]

    def release_canvas_rope_buffers(self):
        try:
            for layer_type, pair in getattr(self, "_canvas_rope_bufs", {}).items():
                for kind, tensor in zip(("cos", "sin"), pair):
                    try:
                        tensor.deallocate(True)
                    except BaseException as cleanup_error:
                        logger.error(f"failed to release canvas RoPE {layer_type}.{kind}: {cleanup_error}")
        finally:
            self._canvas_rope_bufs = {}
            self.use_canvas_rope = False

    # --- Paged-prefix Phase 1 reveal mask (constant-shape written input) ------------
    # A single persistent [1,1,C,p_max+C] additive mask shared by all 30 layers, allocated
    # BEFORE begin_trace_capture and refreshed per block OUTSIDE capture. Phase 1 reveals the
    # committed prefix [0:prompt_len] + all canvas (all-attend, matching today's maskless path)
    # and hides the uncommitted tail [prompt_len:p_max] with NEG. Paired with a fixed p_max
    # prefix read (MutablePrefixKVReader.set_read_span) so the traced graph is shape-invariant
    # → capture-once/replay-many. See doc/optimize_perf/paged_prefix_denoise_design.md §1a/§5.

    def _build_reveal_mask_device(self, prompt_len: int):
        return build_device_canvas_reveal_mask(
            self.tt_model.mesh_device,
            prompt_len=prompt_len,
            canvas_len=self._reveal_canvas_len,
            p_max=self._reveal_p_max,
            layer_type="full_attention",
            enforce_sliding_window=self._reveal_enforce_window,
        )

    def prepare_reveal_mask_buffers(
        self, *, canvas_len: int, p_max: int, prompt_len: int, enforce_window: bool = False
    ):
        """Preallocate the persistent reveal mask OUTSIDE any trace (session-8 rule)."""
        if p_max % ttnn.TILE_SIZE != 0:
            raise ValueError(f"reveal p_max must be tile aligned, got {p_max}")
        self._reveal_canvas_len = int(canvas_len)
        self._reveal_p_max = int(p_max)
        self._reveal_enforce_window = bool(enforce_window)
        if self._reveal_mask_buf is not None:
            self._reveal_mask_buf.deallocate(True)
        self._reveal_mask_buf = self._build_reveal_mask_device(int(prompt_len))
        self.use_reveal_mask = True

    def update_reveal_mask_buffer(self, prompt_len: int):
        """Refresh reveal mask CONTENT for a block's committed ``prompt_len`` (OUTSIDE trace)."""
        if not self.use_reveal_mask:
            return
        if prompt_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"reveal prompt_len must be a 32-tile multiple, got {prompt_len}")
        fresh = self._build_reveal_mask_device(int(prompt_len))
        ttnn.copy(fresh, self._reveal_mask_buf)
        fresh.deallocate(True)

    def _reveal_mask_provider(self, layer_idx):
        return self._reveal_mask_buf

    def release_reveal_mask_buffers(self):
        try:
            if self._reveal_mask_buf is not None:
                self._reveal_mask_buf.deallocate(True)
        except BaseException as cleanup_error:
            logger.error(f"failed to release reveal mask: {cleanup_error}")
        finally:
            self._reveal_mask_buf = None
            self.use_reveal_mask = False

    # --- TP-sharded denoise terminal (DG_TERMINAL_SHARDED) ---------------------------
    #
    # Preallocate the sharded-terminal constants BEFORE ``begin_trace_capture`` (session-8 rule):
    # the per-device vocab index offset and the vocab-row-sharded tied embedding table are
    # persistent constants with fixed device addresses across replays (the KV-cache pattern). Once
    # prepared, ``soft_embedding_signal_sharded`` computes the self-cond signal on the logit shard,
    # and ``denoise_terminal_reductions_sharded`` (module fn) produces the sharded
    # argmax/gumbel/entropy. The per-step logit shard comes from
    # ``denoise_logits_forward(return_sharded=True)``.

    def prepare_sharded_terminal(self, *, canvas_len, vocab_size, embedding_weight=None, embedding_weight_sharded=None):
        """Preallocate the sharded-terminal constants OUTSIDE any trace and enable the sharded path.

        Builds the ``[1,1,canvas_len,TP]`` per-device vocab offset (sharded on tp_axis) and, when a
        tied ``embedding_weight`` (torch ``[vocab, hidden]``) or a prebuilt
        ``embedding_weight_sharded`` is supplied and self-conditioning is active, the vocab-row-
        sharded embedding table. Sets ``self.sharded_terminal``; the offset + table are freed by
        :meth:`release_sharded_terminal`."""
        from models.experimental.diffusion_gemma.tt.sampling import build_vocab_shard_offsets

        tt_model = self.tt_model
        mesh_config = tt_model.mesh_config
        self._sharded_mesh_config = mesh_config
        self._sharded_ccl_manager = tt_model.ccl_manager
        if self._vocab_offsets is not None:
            self._vocab_offsets.deallocate(True)
        self._vocab_offsets = build_vocab_shard_offsets(
            tt_model.mesh_device, mesh_config, canvas_len=canvas_len, vocab_size=vocab_size
        )
        if self._embedding_weight_sharded is not None:
            self._embedding_weight_sharded.deallocate(True)
            self._embedding_weight_sharded = None
        if embedding_weight_sharded is not None:
            self._embedding_weight_sharded = embedding_weight_sharded
        elif embedding_weight is not None and self.self_conditioning is not None:
            self._embedding_weight_sharded = build_self_conditioning_embedding_weight_vocab_sharded(
                tt_model.mesh_device,
                embedding_weight,
                mesh_config,
                hidden_size=self.self_conditioning.hidden_size,
            )
        self.sharded_terminal = True

    def soft_embedding_signal_sharded(self, logits_shard, *, temperature: float = 1.0):
        """Sharded self-conditioning soft-embedding signal ``[1,1,C,hidden]`` from the logit shard.

        Drop-in replacement for ``self_conditioning.soft_embedding`` on the per-device shard;
        requires :meth:`prepare_sharded_terminal` to have built the row-sharded embed table."""
        if self.self_conditioning is None or self._embedding_weight_sharded is None:
            raise RuntimeError("prepare_sharded_terminal must build the row-sharded embedding table first")
        return self.self_conditioning.soft_embedding_sharded(
            logits_shard,
            self._embedding_weight_sharded,
            mesh_config=self._sharded_mesh_config,
            ccl_manager=self._sharded_ccl_manager,
            temperature=temperature,
        )

    def _temperature_at_step(self, step: int) -> float:
        if self.max_denoise_steps is None:
            return 1.0
        from models.experimental.diffusion_gemma.reference.sampling import temperature_at_step

        return temperature_at_step(
            step,
            self.max_denoise_steps,
            self.temperature_start,
            self.temperature_end,
        )

    def release_sharded_terminal(self):
        try:
            for name, tensor in (
                ("vocab_offsets", self._vocab_offsets),
                ("embedding_weight_sharded", self._embedding_weight_sharded),
            ):
                if tensor is not None:
                    try:
                        tensor.deallocate(True)
                    except BaseException as cleanup_error:
                        logger.error(f"failed to release sharded terminal {name}: {cleanup_error}")
        finally:
            self._vocab_offsets = None
            self._embedding_weight_sharded = None
            self.sharded_terminal = False

    def sharded_terminal_context(self):
        """Return the :class:`~...tt.denoise_loop.ShardedTerminalContext` for the sharded
        argmax/gumbel/entropy reductions, or ``None`` for the replicated full-vocab terminal.

        Non-``None`` only when the sharded terminal is prepared (``prepare_sharded_terminal``)
        AND the trace-safe self-cond loop is active — i.e. exactly when :meth:`_trace_safe_call`
        emits a per-device vocab SHARD (``denoise_logits_forward(return_sharded=True)``). Gating
        on both keeps the reductions' input (shard vs full vocab) in lockstep with
        :func:`~...tt.denoise_loop.denoise_step`'s routing, so the eager ``prev_logits`` path (which
        never emits a shard) can't feed full-vocab logits to the sharded ops. The denoise loop /
        trace controllers thread this into ``denoise_step`` (see
        :func:`~...tt.denoise_loop._sharded_terminal_context`)."""
        if not (self.sharded_terminal and self.trace_safe_self_conditioning):
            return None
        if self._vocab_offsets is None:
            return None
        from models.experimental.diffusion_gemma.tt.denoise_loop import ShardedTerminalContext

        return ShardedTerminalContext(
            offsets=self._vocab_offsets,
            mesh_config=self._sharded_mesh_config,
            ccl_manager=self._sharded_ccl_manager,
        )

    def _trace_safe_call(self, canvas_tokens, step: int):
        tt_model = self.tt_model
        read_buf, write_buf = self._signal_read_write_bufs(step)
        canvas_hidden = embed_canvas_tokens(tt_model, canvas_tokens)
        if self.self_conditioning is None:
            conditioned = canvas_hidden
        else:
            # Uniform: forward over the persistent signal read buffer (zeroed for step 0).
            conditioned = self.self_conditioning.forward(canvas_hidden, read_buf)
            canvas_hidden.deallocate(True)
        # DG_TERMINAL_SHARDED: return the per-device vocab shard (skip the ~128 MiB/step
        # full-vocab all-gather); the loop's denoise_step routes its reductions through the
        # sharded ops via sharded_terminal_context(). Default off returns the full replicated
        # logits exactly as before.
        return_sharded = self.sharded_terminal
        logits = denoise_logits_forward(
            tt_model,
            prompt_hidden_by_layer=self.prompt_hidden_by_layer,
            canvas_hidden=conditioned,
            q_rope_offset=self.q_rope_offset,
            prompt_len=self.prompt_len,
            canvas_rope_provider=self._canvas_rope_provider if self.use_canvas_rope else None,
            reveal_mask_provider=self._reveal_mask_provider if self.use_reveal_mask else None,
            return_sharded=return_sharded,
        )
        if conditioned is not canvas_hidden:
            conditioned.deallocate(True)
        if self.self_conditioning is not None:
            # Update the persistent signal buffer in-place for the next step (logits
            # is fully consumed within this step: soft_embedding here + the loop's
            # decision path). Across single-step trace replays the buffer persists.
            # On the sharded terminal the signal is the row-sharded soft-embedding of the
            # logit SHARD (no full-vocab all-gather); otherwise the replicated soft-embedding.
            if return_sharded:
                new_signal = self.soft_embedding_signal_sharded(logits, temperature=self._temperature_at_step(step))
            else:
                new_signal = self.self_conditioning.soft_embedding(
                    logits,
                    self.self_conditioning_embedding_weight,
                    compute_kernel_config=self.self_conditioning_compute_kernel_config,
                    temperature=self._temperature_at_step(step),
                )
            ttnn.copy(new_signal, write_buf)
            new_signal.deallocate(True)
        return logits

    def __call__(self, canvas_tokens, step: int):
        if self.trace_safe_self_conditioning:
            return self._trace_safe_call(canvas_tokens, step)
        old_prev_logits = self.prev_logits
        logits = self.logits_from_tokens(
            self.tt_model,
            prompt_hidden_by_layer=self.prompt_hidden_by_layer,
            canvas_tokens=canvas_tokens,
            self_conditioning=self.self_conditioning,
            prev_logits=old_prev_logits,
            q_rope_offset=self.q_rope_offset,
            prompt_len=self.prompt_len,
            self_conditioning_embedding_weight=self.self_conditioning_embedding_weight,
            self_conditioning_compute_kernel_config=self.self_conditioning_compute_kernel_config,
            self_conditioning_temperature=(self._temperature_at_step(step - 1) if old_prev_logits is not None else 1.0),
        )
        self.prev_logits = logits
        if old_prev_logits is not None:
            old_prev_logits.deallocate(True)
        return logits

    def owns_logits(self, logits) -> bool:
        """Return True when ``logits`` is retained for next-step self-conditioning."""
        return self.prev_logits is logits

    def rebind_prompt(self, prompt_len: int) -> None:
        """Bind a startup-captured reveal-mask trace to a newly prefetched request.

        The model-owned KV cache head has already been overwritten by the request prefill.
        Only persistent buffer contents and scalar position state change here; every address
        baked into the trace remains stable. Rebinding a prefix-shape-baked adapter is rejected
        because it would silently replay another request's prompt.
        """
        if not getattr(self, "use_reveal_mask", False):
            raise RuntimeError("prompt rebind requires a captured DG_DENOISE_REVEAL_MASK adapter")
        resetter = getattr(self.prompt_hidden_by_layer, "reset_prompt_len", None)
        if not callable(resetter):
            raise RuntimeError("prompt rebind requires a MutablePrefixKVReader prefix source")

        prompt_len = int(prompt_len)
        canvas_len = int(getattr(self, "_canvas_rope_len", 0) or 0)
        p_max = int(getattr(self, "_reveal_p_max", 0) or 0)
        if canvas_len and p_max and prompt_len + canvas_len > p_max:
            raise ValueError(
                "request exceeds the up-front capture context: "
                f"{prompt_len} + {canvas_len} = {prompt_len + canvas_len} > {p_max}"
            )

        resetter(prompt_len)
        self.prompt_len = prompt_len
        self.q_rope_offset = prompt_len
        self.update_reveal_mask_buffer(prompt_len)
        self.update_canvas_rope_buffers(prompt_len)

    def advance_prefix_after_commit(self, next_pos: int) -> bool:
        """Expose newly committed KV to later denoise blocks.

        Returns ``True`` for the mutable contiguous-cache reader used by generation
        and serving. Static prompt-hidden test adapters return ``False``.
        """
        if os.environ.get("DG_FROZEN_PREFIX_CONTROL", "0").lower() in ("1", "true", "yes", "on"):
            logger.warning("[DiffusionGemma] DG_FROZEN_PREFIX_CONTROL keeps the initial prompt-only KV span")
            return False
        setter = getattr(self.prompt_hidden_by_layer, "set_prompt_len", None)
        if not callable(setter):
            return False
        setter(next_pos)
        self.prompt_len = int(next_pos)
        self.q_rope_offset = int(next_pos)
        # Reveal-mask capture-once: the read span stays fixed at p_max; growth is exposed by
        # revealing the newly committed prefix in the persistent mask (refreshed OUTSIDE any
        # trace, before the next block's replay). The controller demotes its recapture guard.
        if getattr(self, "use_reveal_mask", False):
            self.update_reveal_mask_buffer(int(next_pos))
        return True

    def reset(self):
        """Release eager and trace-persistent adapter state for request teardown."""
        try:
            if self.prev_logits is not None:
                try:
                    self.prev_logits.deallocate(True)
                except BaseException as cleanup_error:
                    logger.error(f"failed to release previous denoise logits: {cleanup_error}")
            for name in ("signal_buf", "signal_buf_b"):
                tensor = getattr(self, name, None)
                if tensor is not None:
                    try:
                        tensor.deallocate(True)
                    except BaseException as cleanup_error:
                        logger.error(f"failed to release trace self-conditioning {name}: {cleanup_error}")
            self.release_canvas_rope_buffers()
            self.release_reveal_mask_buffers()
            self.release_sharded_terminal()
        finally:
            self.prev_logits = None
            self.signal_buf = None
            self.signal_buf_b = None
            self.trace_safe_self_conditioning = False
            self.signal_ping_pong = False


def make_denoise_logits_adapter_from_kv_cache(
    tt_model,
    *,
    prompt_len: int,
    seq_len_start: int = 0,
    self_conditioning=None,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
    q_rope_offset: int | None = None,
    max_denoise_steps: int | None = None,
    temperature_start: float = 0.8,
    temperature_end: float = 0.4,
    read_prompt_kv_fn=read_prompt_kv_cache_by_layer,
    adapter_cls=DenoiseLogitsAdapter,
):
    """Build a denoise logits adapter from the model's per-layer prompt KV cache."""

    prompt_kv_for_layer = MutablePrefixKVReader(
        tt_model,
        prompt_len=prompt_len,
        seq_len_start=seq_len_start,
        read_fn=read_prompt_kv_fn,
    )

    return adapter_cls(
        tt_model,
        prompt_hidden_by_layer=prompt_kv_for_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=self_conditioning_embedding_weight,
        self_conditioning_compute_kernel_config=self_conditioning_compute_kernel_config,
        q_rope_offset=prompt_len if q_rope_offset is None else q_rope_offset,
        prompt_len=prompt_len,
        max_denoise_steps=max_denoise_steps,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
    )


def make_denoise_logits_adapter_from_checkpoint_state(
    tt_model,
    *,
    prompt_len: int,
    self_conditioning_state,
    embedding_weight,
    config=None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    eps: float | None = None,
    seq_len_start: int = 0,
    q_rope_offset: int | None = None,
    self_conditioning_dtype=ttnn.bfloat16,
    self_conditioning_compute_kernel_config=None,
    max_denoise_steps: int | None = None,
    temperature_start: float = 0.8,
    temperature_end: float = 0.4,
    default_compute_kernel_config_fn=default_self_conditioning_compute_kernel_config,
    self_conditioning_builder=build_self_conditioning,
    embedding_weight_builder=build_self_conditioning_embedding_weight,
    adapter_builder=make_denoise_logits_adapter_from_kv_cache,
):
    """Build the full denoise logits adapter from remapped real-checkpoint pieces."""
    self_conditioning = self_conditioning_builder(
        tt_model.mesh_device,
        self_conditioning_state,
        config=config,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        eps=eps,
        dtype=self_conditioning_dtype,
    )
    if config is not None and hidden_size is None:
        hidden_size = config["hidden_size"] if isinstance(config, dict) else config.hidden_size
    embedding_weight_tt = embedding_weight_builder(
        tt_model.mesh_device,
        embedding_weight,
        hidden_size=hidden_size,
        dtype=self_conditioning_dtype,
    )
    if self_conditioning_compute_kernel_config is None:
        self_conditioning_compute_kernel_config = default_compute_kernel_config_fn()
    return adapter_builder(
        tt_model,
        prompt_len=prompt_len,
        seq_len_start=seq_len_start,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=embedding_weight_tt,
        self_conditioning_compute_kernel_config=self_conditioning_compute_kernel_config,
        q_rope_offset=q_rope_offset,
        max_denoise_steps=max_denoise_steps,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
    )


def make_denoise_logits_adapter_from_remapped_state(
    tt_model,
    *,
    prompt_len: int,
    backbone_state,
    self_conditioning_state,
    embedding_key: str = GEMMA4_LM_PREFIX + "embed_tokens.weight",
    checkpoint_adapter_builder=make_denoise_logits_adapter_from_checkpoint_state,
    **kwargs,
):
    """Build a denoise adapter from ``weight_mapping.remap_state_dict`` outputs."""
    if embedding_key not in backbone_state:
        raise ValueError(f"missing tied embedding weight in backbone_state: {embedding_key}")
    return checkpoint_adapter_builder(
        tt_model,
        prompt_len=prompt_len,
        self_conditioning_state=self_conditioning_state,
        embedding_weight=backbone_state[embedding_key],
        **kwargs,
    )


def make_generation_logits_fn_builder_from_remapped_state(
    *,
    backbone_state,
    self_conditioning_state,
    adapter_builder=make_denoise_logits_adapter_from_remapped_state,
    **adapter_kwargs,
):
    """Return a ``tt.generate`` post-prefill builder for remapped checkpoint state."""

    def logits_fn_builder(
        tt_model,
        *,
        prompt_tokens=None,
        prompt_len: int,
        page_table=None,
        page_tables_per_layer=None,
    ):
        del prompt_tokens, page_table, page_tables_per_layer
        return adapter_builder(
            tt_model,
            prompt_len=prompt_len,
            backbone_state=backbone_state,
            self_conditioning_state=self_conditioning_state,
            **adapter_kwargs,
        )

    return logits_fn_builder


def make_generation_logits_fn_builder_from_checkpoint_state(
    dg_state_dict,
    *,
    remap_fn=remap_state_dict,
    remapped_builder=make_generation_logits_fn_builder_from_remapped_state,
    **adapter_kwargs,
):
    """Return a generation logits builder directly from raw DiffusionGemma state."""
    backbone_state, self_conditioning_state, _ignored_keys = remap_fn(dg_state_dict)
    return remapped_builder(
        backbone_state=backbone_state,
        self_conditioning_state=self_conditioning_state,
        **adapter_kwargs,
    )
