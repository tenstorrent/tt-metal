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


def multi_scale_deformable_attn_uniad_style(
    value_tt,
    value_spatial_shapes,
    sampling_locations_torch,
    attention_weights_torch,
    device,
    num_heads,
    head_dim,
):
    """
    UniAD-style multi-scale deformable attention — batches all sampling points
    per level into a single grid_sample call (vs per-point loop).

    For the decoder (900 queries): 5 grid_sample calls instead of 20.
    Matches UniAD's multi_scale_deformable_attn_pytorch.

    Args:
        value_tt: ttnn [bs, num_keys, num_heads, head_dim] ROW_MAJOR on device
        value_spatial_shapes: torch.Tensor [num_levels, 2]
        sampling_locations_torch: torch [bs, Q, heads, levels, points, 2] in [0,1]
        attention_weights_torch: torch [bs, Q, heads, levels, points]
        device: ttnn device
        num_heads, head_dim: attention config

    Returns:
        ttnn [bs, num_queries, embed_dims] TILE on device
    """
    bs = sampling_locations_torch.shape[0]
    num_queries = sampling_locations_torch.shape[1]
    num_levels = sampling_locations_torch.shape[3]
    num_points = sampling_locations_torch.shape[4]

    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_level_list = ttnn.split(value_tt, split_sizes, dim=1)

    sampling_grids = sampling_locations_torch * 2.0 - 1.0

    output = ttnn.zeros(
        [bs, num_queries, num_heads, head_dim],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)

        # Value: [bs, H*W, heads, dim] → [bs*heads, H, W, dim] (ROW_MAJOR)
        val_l = value_level_list[level]
        val_l = ttnn.permute(val_l, (0, 2, 1, 3))  # [bs, heads, H*W, dim]
        val_l = ttnn.reshape(val_l, (bs * num_heads, H_, W_, head_dim))

        # Grid: batch ALL points together → [bs*heads, Q*points, 1, 2]
        grid = sampling_grids[:, :, :, level, :, :]  # [bs, Q, heads, points, 2]
        grid = grid.permute(0, 2, 1, 3, 4).reshape(bs * num_heads, num_queries * num_points, 1, 2).contiguous()
        grid_tt = ttnn.from_torch(
            grid.to(torch.bfloat16),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        sampled = ttnn.grid_sample(val_l, grid_tt)  # [bs*heads, Q*points, 1, dim]
        ttnn.deallocate(grid_tt)
        ttnn.deallocate(val_l)

        # Reshape to [bs, Q, heads, points, dim]
        sampled = ttnn.squeeze(sampled, 2)  # [bs*heads, Q*points, dim]
        sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
        sampled = ttnn.reshape(sampled, (bs, num_heads, num_queries, num_points, head_dim))
        sampled = ttnn.permute(sampled, (0, 2, 1, 3, 4))  # [bs, Q, heads, points, dim]

        # Attention weights for this level: [bs, Q, heads, points, 1]
        attn = attention_weights_torch[:, :, :, level, :]
        attn = attn.unsqueeze(-1).contiguous()
        attn_tt = ttnn.from_torch(
            attn.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Weighted sum over points
        weighted = ttnn.mul(sampled, attn_tt)
        ttnn.deallocate(sampled)
        ttnn.deallocate(attn_tt)

        level_out = ttnn.sum(weighted, dim=-2)  # [bs, Q, heads, dim]
        ttnn.deallocate(weighted)

        old_output = output
        output = ttnn.add(output, level_out)
        ttnn.deallocate(old_output)
        ttnn.deallocate(level_out)

    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    return output


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
            sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
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

            # Weighted sum over points — all on device
            level_out = None
            for p in range(num_points):
                attn_p = attn_tt[:, :, p : p + 1]
                weighted_p = ttnn.mul(point_chunks[p], attn_p)
                if level_out is None:
                    level_out = weighted_p
                else:
                    old = level_out
                    level_out = ttnn.add(level_out, weighted_p)
                    ttnn.deallocate(old)
                    ttnn.deallocate(weighted_p)

            ttnn.deallocate(attn_tt)
            for pc in point_chunks:
                ttnn.deallocate(pc)

            if chunk_accum is None:
                chunk_accum = level_out
            else:
                old = chunk_accum
                chunk_accum = ttnn.add(chunk_accum, level_out)
                ttnn.deallocate(old)
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

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.permute(output, (1, 0, 2))
    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)

    logger.info("    deform_attn: done.")
    return output


class TtMSDeformAttn:
    """
    Multi-Scale Deformable Attention for DINO encoder/decoder.

    For the encoder (89K queries), 6D intermediate tensors would exceed DRAM
    due to TILE padding on small trailing dims. We project on-device, then
    compute sampling locations on host to avoid the memory blowup.
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
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.params["value_proj"]["weight"], bias=self.params["value_proj"]["bias"])

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
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

        # Offsets and weights still go to host for 6D reshape + softmax
        logger.info("  MSDeformAttn: moving offsets/weights to host...")
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

        logger.info("  MSDeformAttn: computing sampling locations on host...")
        if ref_pts.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1].float(), spatial_shapes[..., 0].float()], dim=-1)
            so_torch = so_torch / offset_normalizer[None, None, None, :, None, :]
            ref_xy = ref_pts.reshape(bs, num_queries, 1, ref_pts.shape[2], 1, 2)
            sampling_locations = ref_xy + so_torch
        elif ref_pts.shape[-1] == 4:
            ref_xy = ref_pts[:, :, None, :, None, :2]
            ref_wh = ref_pts[:, :, None, :, None, 2:]
            sampling_locations = ref_xy + (so_torch / self.num_points) * ref_wh * 0.5
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {ref_pts.shape[-1]}")

        if num_queries <= 2048:
            logger.info(f"  MSDeformAttn: UniAD-style batched grid_sample ({num_queries} queries)...")
            output = multi_scale_deformable_attn_uniad_style(
                value_tt=value,
                value_spatial_shapes=spatial_shapes,
                sampling_locations_torch=sampling_locations,
                attention_weights_torch=aw_torch,
                device=self.device,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
        else:
            logger.info(f"  MSDeformAttn: chunked grid_sample ({num_queries} queries)...")
            output = multi_scale_deformable_attn_ttnn(
                value_tt=value,
                value_spatial_shapes=spatial_shapes,
                sampling_locations_torch=sampling_locations,
                attention_weights_torch=aw_torch,
                device=self.device,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )

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
            logger.info(f"Encoder layer {i} done.")

        return query
