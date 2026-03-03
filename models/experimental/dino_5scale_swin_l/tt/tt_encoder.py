# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the DINO-5scale Deformable DETR Encoder.

Architecture per layer (x6):
  MSDeformAttn (self-attention) -> LayerNorm -> FFN -> LayerNorm

Reuses the multi-scale deformable attention pattern from BEVFormer/UniAD
with native ttnn.grid_sample for bilinear sampling.
"""

import torch
import ttnn
from loguru import logger

UPLOAD_CHUNK_QUERIES = 1024


def decoder_deformable_attn_compute_optimized(
    sampling_offsets_flat,
    attention_weights_flat,
    reference_points,
    spatial_shapes,
    bs,
    num_queries,
    num_heads,
    num_levels,
    num_points,
    device,
):
    # L1 only for small tensors that fit (~1MB); large so_tt/aw_tt/sampling_locations stay in DRAM.
    l1_cfg = ttnn.L1_MEMORY_CONFIG
    dram_cfg = ttnn.DRAM_MEMORY_CONFIG
    so_shape = (bs, num_queries, num_heads, num_levels, num_points, 2)
    aw_shape_flat = (bs, num_queries, num_heads, num_levels * num_points)
    aw_shape_5d = (bs, num_queries, num_heads, num_levels, num_points)

    so_tt = ttnn.reshape(sampling_offsets_flat, so_shape, memory_config=dram_cfg)
    aw_tt = ttnn.reshape(attention_weights_flat, aw_shape_flat, memory_config=dram_cfg)

    aw_tt = ttnn.softmax_in_place(aw_tt)
    aw_tt = ttnn.reshape(aw_tt, aw_shape_5d, memory_config=dram_cfg)

    if isinstance(reference_points, ttnn.Tensor):
        ref_pts = reference_points
    else:
        ref_pts = ttnn.from_torch(reference_points, device=device, memory_config=l1_cfg)

    if ref_pts.shape[-1] == 2:
        spatial_shapes_tt = ttnn.from_torch(
            spatial_shapes.float(), device=device, dtype=ttnn.bfloat16, memory_config=l1_cfg
        )
        offset_normalizer = ttnn.stack([spatial_shapes_tt[..., 1], spatial_shapes_tt[..., 0]], dim=-1)
        offset_normalizer = ttnn.to_memory_config(offset_normalizer, l1_cfg)
        offset_normalizer = ttnn.reshape(offset_normalizer, (1, 1, 1, num_levels, 1, 2))
        so_tt = ttnn.divide(so_tt, offset_normalizer, memory_config=dram_cfg)
        ref_xy = ttnn.reshape(ref_pts, (bs, num_queries, 1, ref_pts.shape[2], 1, 2), memory_config=l1_cfg)
        sampling_locations = ttnn.add(ref_xy, so_tt, memory_config=dram_cfg)
    else:
        ref_xy_4d = ref_pts[:, :, :, :2]
        ref_xy = ttnn.reshape(ref_xy_4d, (bs, num_queries, 1, ref_pts.shape[2], 1, 2), memory_config=l1_cfg)
        if isinstance(ref_xy, torch.Tensor):
            ref_xy = ttnn.from_torch(ref_xy, device=device, memory_config=l1_cfg)
        ref_wh_4d = ref_pts[:, :, :, 2:]
        ref_wh = ttnn.reshape(ref_wh_4d, (bs, num_queries, 1, ref_pts.shape[2], 1, 2), memory_config=l1_cfg)
        if isinstance(ref_wh, torch.Tensor):
            ref_wh = ttnn.from_torch(ref_wh, device=device, memory_config=l1_cfg)

        term1 = ttnn.divide(so_tt, num_points, memory_config=dram_cfg)
        term2 = ttnn.multiply(ref_wh, 0.5, memory_config=l1_cfg)
        offset_term = ttnn.multiply(term1, term2, memory_config=dram_cfg)

        sampling_locations = ttnn.add(ref_xy, offset_term, memory_config=dram_cfg)

    return sampling_locations, aw_tt


# Encoder path (Option A): compute 6D/5D on host, upload this many queries at a time to device.
# Chunk size 1024; only small intermediates (grid ~128KB, attn ~64KB) fit L1, rest in DRAM.


def multi_scale_deformable_attn_ttnn(
    value_tt,
    value_spatial_shapes,
    sampling_locations_tt,
    attention_weights_tt,
    device,
    num_heads,
    head_dim,
):
    """
    Multi-scale deformable attention for a single query chunk. Chunk size 1024.
    Only small intermediates fit L1: grid (~128KB), attn_tt (~64KB), attn_p (~16KB). Rest in DRAM.
    """
    bs = sampling_locations_tt.shape[0]
    chunk_Q = sampling_locations_tt.shape[1]
    num_levels = sampling_locations_tt.shape[3]
    num_points = sampling_locations_tt.shape[4]

    l1_cfg = ttnn.L1_MEMORY_CONFIG
    dram_cfg = ttnn.DRAM_MEMORY_CONFIG

    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_level_list = ttnn.split(value_tt, split_sizes, dim=1)

    value_l_tts = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        val_l = value_level_list[level]
        val_l = ttnn.permute(val_l, (0, 2, 1, 3))
        val_l = ttnn.reshape(val_l, (bs * num_heads, H_, W_, head_dim))
        value_l_tts.append(val_l)

    sampling_grids = ttnn.multiply(sampling_locations_tt, 2.0, memory_config=dram_cfg)
    sampling_grids = ttnn.add(sampling_grids, -1.0, memory_config=dram_cfg)

    chunk_accum = None

    for level in range(num_levels):
        grid = sampling_grids[:, :, :, level, :, :]
        grid = ttnn.permute(grid, (0, 2, 3, 1, 4))
        grid = ttnn.reshape(grid, (bs * num_heads, num_points * chunk_Q, 1, 2))
        grid = ttnn.to_memory_config(grid, l1_cfg)  # ~128KB for chunk_Q=1024, fits L1
        grid = ttnn.to_layout(grid, ttnn.ROW_MAJOR_LAYOUT)

        sampled = ttnn.grid_sample(value_l_tts[level], grid, memory_config=dram_cfg)
        ttnn.deallocate(grid)

        sampled = ttnn.squeeze(sampled, 2)
        sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
        point_chunks = ttnn.split(sampled, [chunk_Q] * num_points, dim=1)
        ttnn.deallocate(sampled)

        attn_tt = attention_weights_tt[:, :, :, level, :]
        attn_tt = ttnn.permute(attn_tt, (0, 2, 1, 3))
        attn_tt = ttnn.reshape(attn_tt, (bs * num_heads, chunk_Q, num_points))
        attn_tt = ttnn.to_memory_config(attn_tt, l1_cfg)  # ~64KB, fits L1

        level_out = None
        for p in range(num_points):
            attn_p = ttnn.slice(attn_tt, [0, 0, p], [bs * num_heads, chunk_Q, p + 1], memory_config=l1_cfg)  # ~16KB
            weighted_p = ttnn.mul(point_chunks[p], attn_p, memory_config=dram_cfg)
            ttnn.deallocate(attn_p)
            if level_out is None:
                level_out = weighted_p
            else:
                level_out = ttnn.add(level_out, weighted_p, memory_config=dram_cfg)
                ttnn.deallocate(weighted_p)

        ttnn.deallocate(attn_tt)
        for pc in point_chunks:
            ttnn.deallocate(pc)

        if chunk_accum is None:
            chunk_accum = level_out
        else:
            chunk_accum = ttnn.add(chunk_accum, level_out, memory_config=dram_cfg)
            ttnn.deallocate(level_out)

    ttnn.deallocate(sampling_grids)
    for vl in value_l_tts:
        ttnn.deallocate(vl)

    output = ttnn.permute(chunk_accum, (1, 0, 2))
    output = ttnn.reshape(output, (bs, chunk_Q, num_heads * head_dim))
    output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
    ttnn.deallocate(chunk_accum)
    return output


def multi_scale_deformable_attn_uniad_style(
    value_tt,
    value_spatial_shapes,
    sampling_locations_tt,
    attention_weights_tt,
    device,
    num_heads,
    head_dim,
):
    """
    Optimized UniAD-style multi-scale deformable attention for decoder.
    L1 only for small tensors that fit (grid ~256KB, attn ~128KB for Q<=2048); rest in DRAM.
    """
    l1_cfg = ttnn.L1_MEMORY_CONFIG
    dram_cfg = ttnn.DRAM_MEMORY_CONFIG
    if isinstance(sampling_locations_tt, torch.Tensor):
        sampling_locations_tt = ttnn.from_torch(sampling_locations_tt, device=device, memory_config=l1_cfg)
    if isinstance(attention_weights_tt, torch.Tensor):
        attention_weights_tt = ttnn.from_torch(attention_weights_tt, device=device, memory_config=l1_cfg)

    bs = sampling_locations_tt.shape[0]
    num_queries = sampling_locations_tt.shape[1]
    num_points = sampling_locations_tt.shape[4]

    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_level_list = ttnn.split(value_tt, split_sizes, dim=1)

    sampling_grids = ttnn.multiply(sampling_locations_tt, 2.0, memory_config=dram_cfg)
    sampling_grids = ttnn.add(sampling_grids, -1.0, memory_config=dram_cfg)

    output = ttnn.zeros(
        [bs, num_queries, num_heads, head_dim],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_cfg,
    )

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)

        # Value: [bs, H*W, heads, dim] → [bs*heads, H, W, dim] (ROW_MAJOR)
        val_l = value_level_list[level]
        val_l = ttnn.permute(val_l, (0, 2, 1, 3))
        val_l = ttnn.reshape(val_l, (bs * num_heads, H_, W_, head_dim))
        val_l = ttnn.to_memory_config(val_l, ttnn.L1_MEMORY_CONFIG)

        grid = sampling_grids[:, :, :, level, :, :]
        grid = ttnn.permute(grid, (0, 2, 1, 3, 4))
        grid = ttnn.reshape(grid, (bs * num_heads, num_queries * num_points, 1, 2))
        # grid = ttnn.to_memory_config(grid, dram_cfg)
        grid = ttnn.to_layout(grid, ttnn.ROW_MAJOR_LAYOUT)

        sampled = ttnn.grid_sample(val_l, grid, memory_config=l1_cfg)
        ttnn.deallocate(grid)
        ttnn.deallocate(val_l)
        # sampled = ttnn.squeeze(sampled, 2)  # [bs*heads, Q*points, dim]
        # sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
        sampled = ttnn.reshape(sampled, (bs, num_heads, num_queries, num_points, head_dim))
        sampled = ttnn.permute(sampled, (0, 2, 1, 3, 4))  # [bs, Q, heads, points, dim]

        # Attention weights for this level: [bs, Q, heads, points, 1] - small, fits L1
        attn = attention_weights_tt[:, :, :, level, :]
        attn = ttnn.unsqueeze(attn, -1)
        # attn = ttnn.to_memory_config(attn, l1_cfg)

        # Weighted sum over points
        weighted = ttnn.mul(sampled, attn, memory_config=l1_cfg)
        ttnn.deallocate(sampled)
        ttnn.deallocate(attn)

        level_out = ttnn.sum(weighted, dim=-2, memory_config=l1_cfg)
        ttnn.deallocate(weighted)

        output = ttnn.add(output, level_out, memory_config=l1_cfg)
        ttnn.deallocate(level_out)

    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    return output


'''
def multi_scale_deformable_attn_ttnn(
    value_tt,
    value_spatial_shapes,
    sampling_locations_torch,
    attention_weights_torch,
    device,
    num_heads,
    head_dim,
    chunk_size=16384,
):
    """
    Multi-scale deformable attention with query chunking.

    All 4 sampling points per level are batched into a single grid_sample
    call, reducing host→device transfers by 4x vs the per-point loop.

    For the encoder (89K queries, 6 chunks, 5 levels): 60 transfers/layer
    instead of 240, totaling 360 across 6 layers (was 1440).

    Args:
        value_tt: ttnn [bs, num_keys, num_heads, head_dim] ROW_MAJOR on device
        value_spatial_shapes: torch.Tensor [num_levels, 2]
        sampling_locations_torch: torch [bs, Q, heads, levels, points, 2]
        attention_weights_torch: torch [bs, Q, heads, levels, points]
        device: ttnn device
        num_heads, head_dim: attention config
        chunk_size: queries per chunk (default 16384)

    Returns:
        ttnn [bs, num_queries, embed_dims] TILE on device
    """
    bs = sampling_locations_torch.shape[0]
    num_queries = sampling_locations_torch.shape[1]
    num_levels = sampling_locations_torch.shape[3]
    num_points = sampling_locations_torch.shape[4]

    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_level_list = ttnn.split(value_tt, split_sizes, dim=1)

    value_l_tts = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        val_l = value_level_list[level]
        val_l = ttnn.permute(val_l, (0, 2, 1, 3))
        val_l = ttnn.reshape(val_l, (bs * num_heads, H_, W_, head_dim))
        value_l_tts.append(val_l)

    sampling_grids = sampling_locations_torch * 2.0 - 1.0

    num_chunks = (num_queries + chunk_size - 1) // chunk_size
    logger.info(
        f"    deform_attn: {num_chunks} chunks, {num_levels} levels, "
        f"{num_points} points/level (batched), "
        f"{num_chunks * num_levels * 2} H→D transfers"
    )

    output_chunks = []

    for c_idx, q_start in enumerate(range(0, num_queries, chunk_size)):
        q_end = min(q_start + chunk_size, num_queries)
        Q = q_end - q_start

        chunk_accum = None

        for level in range(num_levels):
            # --- Transfer 1: grid with all points (points-slow ordering) ---
            grid = sampling_grids[:, q_start:q_end, :, level, :, :]
            grid = grid.permute(0, 2, 3, 1, 4).reshape(bs * num_heads, num_points * Q, 1, 2).contiguous()
            grid_tt = ttnn.from_torch(
                grid.to(torch.bfloat16),
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            # Single grid_sample for all 4 points
            sampled = ttnn.grid_sample(value_l_tts[level], grid_tt)
            ttnn.deallocate(grid_tt)

            sampled = ttnn.squeeze(sampled, 2)
            # sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
            point_chunks = ttnn.split(sampled, [Q] * num_points, dim=1)
            ttnn.deallocate(sampled)

            # --- Transfer 2: attention weights for all points ---
            attn = attention_weights_torch[:, q_start:q_end, :, level, :]
            attn = attn.permute(0, 2, 1, 3).reshape(bs * num_heads, Q, num_points).contiguous()
            attn_tt = ttnn.from_torch(
                attn.to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )

            level_out = None
            for p in range(num_points):
                attn_p = ttnn.slice(attn_tt, [0, 0, p], [bs * num_heads, Q, p + 1])
                weighted_p = ttnn.mul(point_chunks[p], attn_p)
                ttnn.deallocate(attn_p)
                if level_out is None:
                    level_out = weighted_p
                else:
                    # old = level_out
                    level_out = ttnn.add(level_out, weighted_p)
                    # ttnn.deallocate(old)
                    ttnn.deallocate(weighted_p)

            ttnn.deallocate(attn_tt)
            for pc in point_chunks:
                ttnn.deallocate(pc)

            if chunk_accum is None:
                chunk_accum = level_out
            else:
                # old = chunk_accum
                chunk_accum = ttnn.add(chunk_accum, level_out)
                # ttnn.deallocate(old)
                ttnn.deallocate(level_out)

        output_chunks.append(chunk_accum)

    for vl in value_l_tts:
        ttnn.deallocate(vl)

    if len(output_chunks) == 1:
        output = output_chunks[0]
    else:
        output = ttnn.concat(output_chunks, dim=1)
        for c in output_chunks:
            ttnn.deallocate(c)

    # output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.permute(output, (1, 0, 2))
    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)

    logger.info("    deform_attn: done.")
    return output
'''


class TtMSDeformAttn:
    """
    Multi-Scale Deformable Attention for DINO encoder/decoder.

    For the encoder (89K queries), we compute 6D/5D sampling locations and
    attention weights on host, then upload a few query chunks at a time to
    device and run grid computation + attention on device (Option A).
    """

    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        num_levels=5,
        num_points=4,
        batch_first=True,
    ):
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads
        self.batch_first = batch_first
        self.device = device
        self.params = params

    def __call__(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        bs, num_queries, _ = query.shape
        bs, num_keys, _ = value.shape

        logger.info("  MSDeformAttn: linear projections on device...")
        # value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.params["value_proj"]["weight"], bias=self.params["value_proj"]["bias"])

        # query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets_flat = ttnn.linear(
            query, self.params["sampling_offsets"]["weight"], bias=self.params["sampling_offsets"]["bias"]
        )
        attention_weights_flat = ttnn.linear(
            query, self.params["attention_weights"]["weight"], bias=self.params["attention_weights"]["bias"]
        )

        # Option 1: Value reshape on device in ROW_MAJOR (no host transfer)
        logger.info("  MSDeformAttn: value reshape on device (ROW_MAJOR)...")
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.reshape(value, (bs, num_keys, self.num_heads, self.head_dim))

        if key_padding_mask is not None:
            mask = ttnn.reshape(key_padding_mask, (bs, num_keys, 1))
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        if num_queries <= 2048:
            sampling_locations, aw_tt = decoder_deformable_attn_compute_optimized(
                sampling_offsets_flat,
                attention_weights_flat,
                reference_points,
                spatial_shapes,
                bs,
                num_queries,
                self.num_heads,
                self.num_levels,
                self.num_points,
                self.device,
            )
            logger.info(f"  MSDeformAttn: UniAD-style batched grid_sample ({num_queries} queries)...")
            output = multi_scale_deformable_attn_uniad_style(
                value_tt=value,
                value_spatial_shapes=spatial_shapes,
                sampling_locations_tt=sampling_locations,
                attention_weights_tt=aw_tt,
                device=self.device,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
        else:
            # Option A: compute 6D/5D on host, upload few chunks at a time; grid + attention on device.
            logger.info("  MSDeformAttn: computing sampling locations and attention weights on host...")
            so_torch = ttnn.to_torch(sampling_offsets_flat).float()
            aw_torch = ttnn.to_torch(attention_weights_flat).float()

            so_torch = so_torch[:, :num_queries, :].reshape(
                bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2
            )
            aw_torch = aw_torch[:, :num_queries, :].reshape(
                bs, num_queries, self.num_heads, self.num_levels * self.num_points
            )
            aw_torch = torch.softmax(aw_torch, dim=-1)
            aw_torch = aw_torch.reshape(bs, num_queries, self.num_heads, self.num_levels, self.num_points)

            if isinstance(reference_points, ttnn.Tensor):
                ref_pts = ttnn.to_torch(reference_points).float()
            else:
                ref_pts = reference_points.float()

            if ref_pts.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1].float(), spatial_shapes[..., 0].float()], dim=-1
                )
                so_torch = so_torch / offset_normalizer[None, None, None, :, None, :]
                ref_xy = ref_pts.reshape(bs, num_queries, 1, ref_pts.shape[2], 1, 2)
                sampling_locations_torch = ref_xy + so_torch
            elif ref_pts.shape[-1] == 4:
                ref_xy = ref_pts[:, :, None, :, None, :2]
                ref_wh = ref_pts[:, :, None, :, None, 2:]
                sampling_locations_torch = ref_xy + (so_torch / self.num_points) * ref_wh * 0.5
            else:
                raise ValueError(f"reference_points last dim must be 2 or 4, got {ref_pts.shape[-1]}")

            # Upload full sampling_locations and attention_weights to device once; slice per chunk in loop.
            sampling_locations_tt = ttnn.from_torch(
                sampling_locations_torch.to(torch.bfloat16),
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            attention_weights_tt = ttnn.from_torch(
                aw_torch.to(torch.bfloat16),
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            num_upload_chunks = (num_queries + UPLOAD_CHUNK_QUERIES - 1) // UPLOAD_CHUNK_QUERIES
            logger.info(
                f"  MSDeformAttn: {num_upload_chunks} chunks (chunk={UPLOAD_CHUNK_QUERIES}), "
                f"grid+attention on device ({num_queries} queries)..."
            )
            output_chunks = []
            for q_start in range(0, num_queries, UPLOAD_CHUNK_QUERIES):
                q_end = min(q_start + UPLOAD_CHUNK_QUERIES, num_queries)
                loc_chunk_tt = ttnn.slice(
                    sampling_locations_tt,
                    [0, q_start, 0, 0, 0, 0],
                    [bs, q_end, self.num_heads, self.num_levels, self.num_points, 2],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                aw_chunk_tt = ttnn.slice(
                    attention_weights_tt,
                    [0, q_start, 0, 0, 0],
                    [bs, q_end, self.num_heads, self.num_levels, self.num_points],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                chunk_out = multi_scale_deformable_attn_ttnn(
                    value_tt=value,
                    value_spatial_shapes=spatial_shapes,
                    sampling_locations_tt=loc_chunk_tt,
                    attention_weights_tt=aw_chunk_tt,
                    device=self.device,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                )
                ttnn.deallocate(loc_chunk_tt)
                ttnn.deallocate(aw_chunk_tt)
                output_chunks.append(chunk_out)

            ttnn.deallocate(sampling_locations_tt)
            ttnn.deallocate(attention_weights_tt)

            if len(output_chunks) == 1:
                output = output_chunks[0]
            else:
                output = ttnn.concat(output_chunks, dim=1)
                for c in output_chunks:
                    ttnn.deallocate(c)
        logger.info("  MSDeformAttn: output projection on device...")
        output = ttnn.linear(output, self.params["output_proj"]["weight"], bias=self.params["output_proj"]["bias"])

        output = ttnn.add(output, identity)
        logger.info("  MSDeformAttn: done.")
        return output


class TtFFN:
    """Feed-forward network: Linear -> ReLU -> Linear + residual."""

    def __init__(self, params, device):
        self.device = device
        self.w1 = params["fc1"]["weight"]
        self.b1 = params["fc1"]["bias"]
        self.w2 = params["fc2"]["weight"]
        self.b2 = params["fc2"]["bias"]

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x
        x = ttnn.linear(x, self.w1, bias=self.b1)
        x = ttnn.relu(x)
        x = ttnn.linear(x, self.w2, bias=self.b2)
        return ttnn.add(x, identity)


class TtDINOEncoderLayer:
    """Single encoder layer: MSDeformAttn -> LN -> FFN -> LN."""

    def __init__(self, params, device, embed_dims=256, num_heads=8, num_levels=5, num_points=4):
        self.self_attn = TtMSDeformAttn(
            params["self_attn"],
            device,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        self.ffn = TtFFN(params["ffn"], device)
        self.norm1_w = params["norms"][0]["weight"]
        self.norm1_b = params["norms"][0]["bias"]
        self.norm2_w = params["norms"][1]["weight"]
        self.norm2_b = params["norms"][1]["bias"]

    def __call__(
        self,
        query,
        query_pos,
        key_padding_mask,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        reference_points,
        **kwargs,
    ):
        query = self.self_attn(
            query=query,
            value=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = ttnn.layer_norm(query, weight=self.norm1_w, bias=self.norm1_b)
        query = self.ffn(query)
        query = ttnn.layer_norm(query, weight=self.norm2_w, bias=self.norm2_b)
        return query


class TtDINOEncoder:
    """
    DINO Deformable DETR Encoder (6 layers).

    Input:  flattened multi-scale features [B, sum(H_i*W_i), 256]
    Output: encoder memory [B, sum(H_i*W_i), 256]
    """

    def __init__(self, params, device, num_layers=6, embed_dims=256, num_heads=8, num_levels=5, num_points=4):
        self.device = device
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.layers = [
            TtDINOEncoderLayer(
                params["layers"][i],
                device,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
            )
            for i in range(num_layers)
        ]

    @staticmethod
    def get_encoder_reference_points(spatial_shapes, valid_ratios, device):
        """
        Generate 2D reference points for encoder self-attention.

        Returns: [B, sum(H_i*W_i), num_levels, 2] on ttnn device.
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32),
                indexing="ij",
            )
            vr = valid_ratios[:, lvl]  # [B, 2]
            ref_y = ref_y.reshape(-1)[None] / (vr[:, 1:2] * H)
            ref_x = ref_x.reshape(-1)[None] / (vr[:, 0:1] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [B, N, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [B, N, num_levels, 2]
        return reference_points

    def __call__(
        self,
        feat,
        feat_pos,
        feat_mask,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    ):
        """
        Args:
            feat: [B, N, 256] flattened multi-scale features (ttnn tensor)
            feat_pos: [B, N, 256] positional encoding + level embed (ttnn tensor)
            feat_mask: [B, N] padding mask or None
            spatial_shapes: torch.Tensor [num_levels, 2]
            level_start_index: torch.Tensor [num_levels]
            valid_ratios: torch.Tensor [B, num_levels, 2]

        Returns:
            memory: [B, N, 256] encoder output (ttnn tensor)
        """
        logger.info("Computing encoder reference points...")
        reference_points = self.get_encoder_reference_points(spatial_shapes, valid_ratios, device=self.device)
        logger.info(f"Reference points shape: {reference_points.shape}")

        query = feat
        for i, layer in enumerate(self.layers):
            logger.info(f"Encoder layer {i} starting...")
            query = layer(
                query=query,
                query_pos=feat_pos,
                key_padding_mask=feat_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
            )
            ttnn.ReadDeviceProfiler(self.device)
            logger.info(f"Encoder layer {i} done.")

        return query
