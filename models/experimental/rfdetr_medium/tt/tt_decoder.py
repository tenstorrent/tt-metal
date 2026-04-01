# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Transformer Decoder for RF-DETR Medium.
Pure TTNN — all computation on device including deformable cross-attention.

4 decoder layers, each:
  1. Self-attention: 8 heads, d_model=256
  2. Deformable cross-attention: 16 heads, 1 level (P4), 2 sampling points
     Implemented with ttnn.grid_sample (BEVFormer pattern)
  3. FFN: 256 → 2048 → 256 (ReLU)
  4. Post-LayerNorm after each sub-layer

Deformable attention pattern from:
  models/experimental/bevformer/tt/tt_ms_deformable_attention.py
"""

import math
import torch
import ttnn

from models.experimental.rfdetr_medium.common import (
    HIDDEN_DIM,
    SA_NHEADS,
    CA_NHEADS,
    DEC_N_POINTS,
    DEC_LAYERS,
    BBOX_REPARAM,
    LITE_REFPOINT_REFINE,
)

try:
    _CoreGrid = ttnn.CoreGrid
except AttributeError:
    _CoreGrid = ttnn.types.CoreGrid
CORE_GRID = _CoreGrid(y=8, x=12)
CA_HEAD_DIM = HIDDEN_DIM // CA_NHEADS  # 256 / 16 = 16


# ---------------------------------------------------------------------------
# Sine positional embedding (pure TTNN, on device)
# Uses ttnn.sin, ttnn.cos, ttnn.arange — same pattern as pi0/tt/ttnn_common.py
# ---------------------------------------------------------------------------


def gen_sineembed_for_position(pos_tensor, dim, device):
    """
    Sine positional embeddings for reference points. Pure TTNN on device.

    Args:
        pos_tensor: [B, N, 2] or [B, N, 4] TTNN tensor of coordinates in [0,1]
        dim: half-embedding dimension (typically HIDDEN_DIM // 2 = 128)
        device: TTNN device

    Returns:
        [B, N, dim*2] (for 2D) or [B, N, dim*4] (for 4D) TTNN tensor on device
    """
    scale = 2 * math.pi

    # dim_t: [1, 1, dim] frequency divisors = 10000^(2*(i//2)/dim) for i in [0..dim-1]
    dim_t_torch = torch.arange(dim, dtype=torch.float32)
    dim_t_torch = 10000.0 ** (2 * (dim_t_torch // 2) / dim)
    dim_t_torch = dim_t_torch.reshape(1, 1, dim)
    dim_t = ttnn.from_torch(dim_t_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _embed_coord(coord_idx):
        """Embed a single coordinate channel → [B, N, dim] with interleaved sin/cos."""
        coord = pos_tensor[:, :, coord_idx : coord_idx + 1]  # [B, N, 1]
        coord = ttnn.mul(coord, scale)

        pos_vals = ttnn.div(coord, dim_t)  # [B, N, dim] (broadcast)
        sin_vals = ttnn.sin(pos_vals)
        cos_vals = ttnn.cos(pos_vals)

        # PyTorch ref does: stack(sin[0::2], cos[1::2], dim=3).flatten(2)
        # dim_t pairs (2k, 2k+1) have equal values, so sin/cos at those positions
        # are identical. We pick sin at even positions, cos at odd to match exactly.
        half = dim // 2
        sin_reshaped = ttnn.reshape(sin_vals, (sin_vals.shape[0], sin_vals.shape[1], half, 2))
        cos_reshaped = ttnn.reshape(cos_vals, (cos_vals.shape[0], cos_vals.shape[1], half, 2))

        sin_even = sin_reshaped[:, :, :, :1]  # [B, N, half, 1]
        cos_odd = cos_reshaped[:, :, :, 1:]  # [B, N, half, 1]
        interleaved = ttnn.concat([sin_even, cos_odd], dim=-1)  # [B, N, half, 2]
        return ttnn.reshape(interleaved, (interleaved.shape[0], interleaved.shape[1], dim))

    # Build embeddings for each coordinate
    pos_y = _embed_coord(1)  # y first (matches PyTorch: pos_y, pos_x order)
    pos_x = _embed_coord(0)

    num_coords = pos_tensor.shape[-1]
    if num_coords == 2:
        return ttnn.concat([pos_y, pos_x], dim=-1)  # [B, N, dim*2]
    elif num_coords == 4:
        pos_w = _embed_coord(2)
        pos_h = _embed_coord(3)
        return ttnn.concat([pos_y, pos_x, pos_w, pos_h], dim=-1)  # [B, N, dim*4]
    else:
        raise ValueError(f"Expected 2 or 4 coordinates, got {num_coords}")


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------


def decoder_self_attention(tgt, query_pos, params):
    """
    Decoder self-attention. 8 heads, d_model=256. All on device.
    """
    q = ttnn.add(tgt, query_pos)
    k = ttnn.add(tgt, query_pos)

    head_size = HIDDEN_DIM // SA_NHEADS
    scale = 1.0 / math.sqrt(head_size)

    q_proj = ttnn.linear(
        q,
        params["q_proj_weight"],
        bias=params["q_proj_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    k_proj = ttnn.linear(
        k,
        params["k_proj_weight"],
        bias=params["k_proj_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    v_proj = ttnn.linear(
        tgt,
        params["v_proj_weight"],
        bias=params["v_proj_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )

    B, N, _ = q_proj.shape
    q_proj = ttnn.reshape(q_proj, (B, N, SA_NHEADS, head_size))
    q_proj = ttnn.permute(q_proj, (0, 2, 1, 3))
    k_proj = ttnn.reshape(k_proj, (B, N, SA_NHEADS, head_size))
    k_proj = ttnn.permute(k_proj, (0, 2, 1, 3))
    v_proj = ttnn.reshape(v_proj, (B, N, SA_NHEADS, head_size))
    v_proj = ttnn.permute(v_proj, (0, 2, 1, 3))

    q_proj = ttnn.mul_(q_proj, scale)
    attn = ttnn.matmul(
        q_proj, ttnn.permute(k_proj, (0, 1, 3, 2)), memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    attn = ttnn.softmax_in_place(attn, numeric_stable=True)
    out = ttnn.matmul(attn, v_proj, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

    out = ttnn.permute(out, (0, 2, 1, 3))
    out = ttnn.reshape(out, (B, N, HIDDEN_DIM))
    out = ttnn.linear(
        out,
        params["out_proj_weight"],
        bias=params["out_proj_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )

    tgt = ttnn.add(tgt, out)
    tgt = ttnn.layer_norm(
        tgt,
        weight=params["norm1_weight"],
        bias=params["norm1_bias"],
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return tgt


# ---------------------------------------------------------------------------
# Deformable cross-attention (on device, BEVFormer pattern)
# ---------------------------------------------------------------------------


def single_level_deformable_attn(value, spatial_shape, sampling_locations, attention_weights, device):
    """
    Core single-level deformable attention using ttnn.grid_sample.

    Args:
        value: [B, H*W, num_heads, head_dim] on device
        spatial_shape: (H, W) tuple
        sampling_locations: [B, num_queries, num_heads, num_points, 2] on device (in [0,1])
        attention_weights: [B, num_queries, num_heads, num_points] on device (softmaxed)

    Returns:
        [B, num_queries, embed_dims] on device
    """
    bs = value.shape[0]
    num_queries = sampling_locations.shape[1]
    H, W = spatial_shape

    # value: [B, H*W, num_heads, head_dim] → [B*num_heads, H, W, head_dim]
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.permute(value, (0, 2, 1, 3))  # [B, num_heads, H*W, head_dim]
    value = ttnn.reshape(value, (bs * CA_NHEADS, H, W, CA_HEAD_DIM))

    # grid_sample requires last dim divisible by 32; head_dim=16, so pad to 32
    PADDED_HEAD_DIM = 32
    if CA_HEAD_DIM < PADDED_HEAD_DIM:
        pad_size = PADDED_HEAD_DIM - CA_HEAD_DIM
        zero_pad = ttnn.from_torch(
            torch.zeros(bs * CA_NHEADS, H, W, pad_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        value = ttnn.concat([value, zero_pad], dim=-1)  # [B*heads, H, W, 32]

    # sampling_locations [0,1] → [-1,1] for grid_sample
    sampling_grid = ttnn.mul(sampling_locations, 2.0)
    sampling_grid = ttnn.sub(sampling_grid, 1.0)

    # [B, num_queries, num_heads, num_points, 2] → [B*num_heads, num_queries*num_points, 1, 2]
    sampling_grid = ttnn.to_layout(sampling_grid, layout=ttnn.ROW_MAJOR_LAYOUT)
    sampling_grid = ttnn.permute(sampling_grid, (0, 2, 1, 3, 4))  # [B, num_heads, Q, P, 2]
    sampling_grid = ttnn.reshape(sampling_grid, (bs * CA_NHEADS, num_queries * DEC_N_POINTS, 1, 2))

    # Bilinear sampling: [B*num_heads, Q*P, 1, padded_head_dim]
    sampled = ttnn.grid_sample(value, sampling_grid)

    # Slice back to actual head_dim if we padded
    if CA_HEAD_DIM < PADDED_HEAD_DIM:
        sampled = sampled[:, :, :, :CA_HEAD_DIM]

    # [B*num_heads, Q*P, 1, head_dim] → [B*num_heads, head_dim, Q, P]
    sampled = ttnn.squeeze(sampled, 2)  # [B*num_heads, Q*P, head_dim]
    sampled = ttnn.reshape(sampled, (bs * CA_NHEADS, num_queries, DEC_N_POINTS, CA_HEAD_DIM))
    sampled = ttnn.permute(sampled, (0, 3, 1, 2))  # [B*num_heads, head_dim, Q, P]

    # attention_weights: [B, Q, num_heads, P] → [B*num_heads, 1, Q, P]
    attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3))  # [B, num_heads, Q, P]
    attention_weights = ttnn.reshape(attention_weights, (bs * CA_NHEADS, 1, num_queries, DEC_N_POINTS))

    # Weighted sum over sampling points
    output = ttnn.mul(sampled, attention_weights)
    output = ttnn.sum(output, dim=-1)  # [B*num_heads, head_dim, Q]

    # Reshape to [B, Q, embed_dims]
    output = ttnn.reshape(output, (bs, CA_NHEADS * CA_HEAD_DIM, num_queries))
    output = ttnn.permute(output, (0, 2, 1))  # [B, Q, embed_dims]

    return output


def decoder_cross_attention(tgt, query_pos, memory, reference_points_4d, spatial_shape, params, device):
    """
    Deformable cross-attention. Pure TTNN using grid_sample.

    Single level (P4), 16 heads, 2 sampling points.
    Uses 4D reference points [cx, cy, w, h] with the MSDeformAttn formula:
      sampling_locs = ref_center + offsets / n_points * ref_wh * 0.5

    Args:
        tgt: [B, 300, 256] query features on device
        query_pos: [B, 300, 256] positional encoding on device
        memory: [B, 1296, 256] encoder features on device
        reference_points_4d: [B, 300, 4] reference points [cx,cy,w,h] on device
        spatial_shape: (36, 36) tuple
        params: cross-attention parameters dict

    Returns:
        [B, 300, 256] on device
    """
    identity = tgt
    query = ttnn.add(tgt, query_pos)

    bs = query.shape[0]
    num_queries = query.shape[1]
    num_keys = memory.shape[1]

    # Value projection: [B, 1296, 256] → [B, 1296, 16, 16]
    value = ttnn.linear(
        memory,
        params["ca_value_weight"],
        bias=params["ca_value_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    value = ttnn.reshape(value, (bs, num_keys, CA_NHEADS, CA_HEAD_DIM))

    # Sampling offsets: [B, 300, 256] → [B, 300, 16*1*2*2] = [B, 300, 64]
    sampling_offsets = ttnn.linear(
        query,
        params["ca_offset_weight"],
        bias=params["ca_offset_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    sampling_offsets = ttnn.reshape(sampling_offsets, (bs, num_queries, CA_NHEADS, DEC_N_POINTS, 2))

    # Attention weights: [B, 300, 256] → [B, 300, 16*1*2] = [B, 300, 32]
    attn_weights = ttnn.linear(
        query,
        params["ca_attn_weight"],
        bias=params["ca_attn_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    attn_weights = ttnn.reshape(attn_weights, (bs, num_queries, CA_NHEADS, DEC_N_POINTS))
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    # 4D formula: sampling_locs = ref_center + offsets / n_points * ref_wh * 0.5
    ref_center = reference_points_4d[:, :, :2]  # [B, Q, 2]
    ref_wh = reference_points_4d[:, :, 2:]  # [B, Q, 2]
    ref_center = ttnn.reshape(ref_center, (bs, num_queries, 1, 1, 2))
    ref_wh = ttnn.reshape(ref_wh, (bs, num_queries, 1, 1, 2))

    # offsets / n_points * ref_wh * 0.5
    scaled_offsets = ttnn.mul(sampling_offsets, 0.5 / DEC_N_POINTS)
    scaled_offsets = ttnn.mul(scaled_offsets, ref_wh)
    sampling_locations = ttnn.add(ref_center, scaled_offsets)

    # Run deformable attention core
    output = single_level_deformable_attn(value, spatial_shape, sampling_locations, attn_weights, device)

    # Output projection
    output = ttnn.linear(
        output,
        params["ca_output_weight"],
        bias=params["ca_output_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )

    # Residual + LayerNorm
    tgt = ttnn.add(identity, output)
    tgt = ttnn.layer_norm(
        tgt,
        weight=params["norm2_weight"],
        bias=params["norm2_bias"],
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return tgt


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------


def decoder_ffn(tgt, params):
    """Decoder FFN: Linear(256→2048) → ReLU → Linear(2048→256). On device."""
    ff = ttnn.linear(
        tgt,
        params["linear1_weight"],
        bias=params["linear1_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    ff = ttnn.relu(ff)
    ff = ttnn.linear(
        ff,
        params["linear2_weight"],
        bias=params["linear2_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )

    tgt = ttnn.add(tgt, ff)
    tgt = ttnn.layer_norm(
        tgt,
        weight=params["norm3_weight"],
        bias=params["norm3_bias"],
        epsilon=1e-05,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return tgt


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------


def ref_point_head_forward(sine_embed, decoder_params, device):
    """
    ref_point_head MLP on device: Linear(512, 256) + ReLU + Linear(256, 256).
    """
    x = ttnn.linear(
        sine_embed,
        decoder_params["ref_point_head_w0"],
        bias=decoder_params["ref_point_head_b0"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    x = ttnn.relu(x)
    x = ttnn.linear(
        x,
        decoder_params["ref_point_head_w1"],
        bias=decoder_params["ref_point_head_b1"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
    )
    return x


def decoder_forward(
    tgt, memory, refpoints_unsigmoid, spatial_shapes, level_start_index, valid_ratios, decoder_params, device
):
    """
    Full decoder: 4 layers of self-attn → deformable cross-attn → FFN.
    Pure TTNN — all computation on device including sine embedding and ref_point_head MLP.

    Args:
        tgt: [B, 300, 256] initial queries on device
        memory: [B, 1296, 256] encoder features on device
        refpoints_unsigmoid: [B, 300, 4] reference points on device (TTNN tensor)
        spatial_shapes: torch.Tensor [1, 2] = [[36, 36]]
        level_start_index: torch.Tensor [1] = [0]
        valid_ratios: torch.Tensor or None
        decoder_params: parameter dict (all TTNN tensors, no PyTorch modules)
        device: TTNN device

    Returns:
        intermediates: list of [B, 300, 256] decoder outputs
        refpoints: reference points on device
    """
    output = tgt
    intermediates = []

    spatial_shape = (int(spatial_shapes[0, 0].item()), int(spatial_shapes[0, 1].item()))

    for layer_idx in range(DEC_LAYERS):
        # Extract 4D reference points for cross-attention
        if BBOX_REPARAM:
            ref_4d = refpoints_unsigmoid[:, :, :4]  # [B, Q, 4] raw cx,cy,w,h
        else:
            ref_4d = ttnn.sigmoid(refpoints_unsigmoid)[:, :, :4]

        # Query positional encoding: sine embedding → ref_point_head MLP (all on device)
        if LITE_REFPOINT_REFINE and layer_idx == 0 or not LITE_REFPOINT_REFINE:
            query_sine = gen_sineembed_for_position(ref_4d, HIDDEN_DIM // 2, device)  # [B, Q, 512]
            query_pos = ref_point_head_forward(query_sine, decoder_params, device)  # [B, Q, 256]

        # Self-attention
        output = decoder_self_attention(output, query_pos, decoder_params["layers"][layer_idx])

        # Deformable cross-attention with 4D reference points (on device)
        output = decoder_cross_attention(
            output,
            query_pos,
            memory,
            ref_4d,
            spatial_shape,
            decoder_params["layers"][layer_idx],
            device,
        )

        # FFN
        output = decoder_ffn(output, decoder_params["layers"][layer_idx])

        intermediates.append(output)

    # Final norm
    if decoder_params.get("final_norm_weight") is not None:
        output = ttnn.layer_norm(
            output,
            weight=decoder_params["final_norm_weight"],
            bias=decoder_params["final_norm_bias"],
            epsilon=1e-05,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        intermediates[-1] = output

    return intermediates, refpoints_unsigmoid
