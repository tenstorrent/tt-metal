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

UPLOAD_CHUNK_QUERIES = 2048


def compute_sampling_locations_and_attention_weights(
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
    spatial_shapes_tt=None,
):
    so_shape = (bs, num_queries, num_heads, num_levels, num_points, 2)
    aw_shape_flat = (bs, num_queries, num_heads, num_levels * num_points)
    aw_shape_5d = (bs, num_queries, num_heads, num_levels, num_points)

    so_tt = ttnn.reshape(sampling_offsets_flat, so_shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    aw_tt = ttnn.reshape(attention_weights_flat, aw_shape_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    aw_tt = ttnn.softmax_in_place(aw_tt)
    aw_tt = ttnn.reshape(aw_tt, aw_shape_5d, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if isinstance(reference_points, ttnn.Tensor):
        ref_pts = reference_points
    else:
        ref_pts = ttnn.from_torch(reference_points, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if ref_pts.shape[-1] == 2:
        if spatial_shapes_tt is None:
            spatial_shapes_tt = ttnn.from_torch(
                spatial_shapes.float(), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            spatial_shapes_tt = ttnn.to_memory_config(spatial_shapes_tt, ttnn.L1_MEMORY_CONFIG)
        offset_normalizer = ttnn.stack([spatial_shapes_tt[..., 1], spatial_shapes_tt[..., 0]], dim=-1)
        offset_normalizer = ttnn.to_memory_config(offset_normalizer, ttnn.L1_MEMORY_CONFIG)
        offset_normalizer = ttnn.reshape(offset_normalizer, (1, 1, 1, num_levels, 1, 2))
        ttnn.divide_(so_tt, offset_normalizer)
        ttnn.deallocate(offset_normalizer)
        ref_xy = ttnn.reshape(
            ref_pts, (bs, num_queries, 1, ref_pts.shape[2], 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.add_(so_tt, ref_xy)
        ttnn.deallocate(ref_xy)
        sampling_locations = so_tt
    else:
        ref_xy_4d = ref_pts[:, :, :, :2]
        ref_xy = ttnn.reshape(
            ref_xy_4d, (bs, num_queries, 1, ref_pts.shape[2], 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if isinstance(ref_xy, torch.Tensor):
            ref_xy = ttnn.from_torch(ref_xy, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ref_wh_4d = ref_pts[:, :, :, 2:]
        ref_wh = ttnn.reshape(
            ref_wh_4d, (bs, num_queries, 1, ref_pts.shape[2], 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if isinstance(ref_wh, torch.Tensor):
            ref_wh = ttnn.from_torch(ref_wh, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.divide_(so_tt, num_points)
        term2 = ttnn.multiply(ref_wh, 0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ref_wh)
        offset_term = ttnn.multiply(so_tt, term2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(so_tt)
        ttnn.deallocate(term2)

        ttnn.add_(offset_term, ref_xy)
        ttnn.deallocate(ref_xy)
        sampling_locations = offset_term

    return sampling_locations, aw_tt


def split_value_into_levels(value_tt, value_spatial_shapes, num_heads, head_dim):
    bs = value_tt.shape[0]
    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_level_list = ttnn.split(value_tt, split_sizes, dim=1)
    value_l_tts = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        val_l = value_level_list[level]
        val_l = ttnn.permute(val_l, (0, 2, 1, 3))
        val_l = ttnn.reshape(val_l, (bs * num_heads, H_, W_, head_dim))
        val_l = ttnn.to_layout(val_l, ttnn.ROW_MAJOR_LAYOUT)
        val_l = ttnn.to_memory_config(val_l, ttnn.DRAM_MEMORY_CONFIG)
        value_l_tts.append(val_l)
    return value_l_tts


def multi_scale_deformable_attn_ttnn(
    value_l_tts,
    sampling_locations_tt,
    attention_weights_tt,
    device,
    num_heads,
    head_dim,
):
    bs = sampling_locations_tt.shape[0]
    chunk_Q = sampling_locations_tt.shape[1]
    num_levels = sampling_locations_tt.shape[3]
    num_points = sampling_locations_tt.shape[4]

    sampling_grids = ttnn.multiply(sampling_locations_tt, 2.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sampling_grids = ttnn.add(sampling_grids, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    chunk_accum = None

    for level in range(num_levels):
        grid = sampling_grids[:, :, :, level, :, :]
        grid = ttnn.permute(grid, (0, 2, 3, 1, 4))
        grid = ttnn.reshape(grid, (bs * num_heads, num_points * chunk_Q, 1, 2))
        grid = ttnn.to_memory_config(grid, ttnn.DRAM_MEMORY_CONFIG)
        grid = ttnn.to_layout(grid, ttnn.ROW_MAJOR_LAYOUT)

        sampled = ttnn.grid_sample(value_l_tts[level], grid, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(grid)

        sampled = ttnn.squeeze(sampled, 2)
        sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
        point_chunks = ttnn.split(sampled, [chunk_Q] * num_points, dim=1)
        ttnn.deallocate(sampled)

        attn_tt = attention_weights_tt[:, :, :, level, :]
        attn_tt = ttnn.permute(attn_tt, (0, 2, 1, 3))
        attn_tt = ttnn.reshape(attn_tt, (bs * num_heads, chunk_Q, num_points))
        attn_tt = ttnn.to_memory_config(attn_tt, ttnn.L1_MEMORY_CONFIG)

        level_out = None
        for p in range(num_points):
            attn_p = ttnn.slice(
                attn_tt, [0, 0, p], [bs * num_heads, chunk_Q, p + 1], memory_config=ttnn.L1_MEMORY_CONFIG
            )
            weighted_p = ttnn.mul(point_chunks[p], attn_p, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_p)
            if level_out is None:
                level_out = weighted_p
            else:
                level_out = ttnn.add(level_out, weighted_p, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(weighted_p)

        ttnn.deallocate(attn_tt)
        for pc in point_chunks:
            ttnn.deallocate(pc)

        if chunk_accum is None:
            chunk_accum = level_out
        else:
            chunk_accum = ttnn.add(chunk_accum, level_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(level_out)

    ttnn.deallocate(sampling_grids)

    output = ttnn.permute(chunk_accum, (1, 0, 2))
    output = ttnn.reshape(output, (bs, chunk_Q, num_heads * head_dim))
    output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
    ttnn.deallocate(chunk_accum)
    return output


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
        trace_mode=False,
    ):
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads
        self.batch_first = batch_first
        self.device = device
        self.params = params
        self.trace_mode = trace_mode

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
        spatial_shapes_tt=None,
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

        value = ttnn.linear(value, self.params["value_proj"]["weight"], bias=self.params["value_proj"]["bias"])
        sampling_offsets_flat = ttnn.linear(
            query, self.params["sampling_offsets"]["weight"], bias=self.params["sampling_offsets"]["bias"]
        )
        attention_weights_flat = ttnn.linear(
            query, self.params["attention_weights"]["weight"], bias=self.params["attention_weights"]["bias"]
        )

        logger.info("  MSDeformAttn: value reshape on device (ROW_MAJOR)...")
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.reshape(value, (bs, num_keys, self.num_heads, self.head_dim))

        if key_padding_mask is not None:
            mask = ttnn.reshape(key_padding_mask, (bs, num_keys, 1))
            value = ttnn.where(mask, ttnn.multiply(value, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG), value)

        if self.trace_mode and spatial_shapes_tt is not None and isinstance(reference_points, ttnn.Tensor):
            logger.info("  MSDeformAttn: device-only path (trace_mode, per-chunk)...")
            value_l_tts = split_value_into_levels(value, spatial_shapes, self.num_heads, self.head_dim)

            output_chunks_device = []
            for q_start in range(0, num_queries, UPLOAD_CHUNK_QUERIES):
                q_end = min(q_start + UPLOAD_CHUNK_QUERIES, num_queries)
                chunk_Q = q_end - q_start
                so_chunk = ttnn.slice(
                    sampling_offsets_flat,
                    [0, q_start, 0],
                    [bs, q_end, self.num_heads * self.num_levels * self.num_points * 2],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                aw_chunk = ttnn.slice(
                    attention_weights_flat,
                    [0, q_start, 0],
                    [bs, q_end, self.num_heads * self.num_levels * self.num_points],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ref_chunk = ttnn.slice(
                    reference_points,
                    [0, q_start, 0, 0],
                    [bs, q_end, self.num_levels, 2],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                loc_chunk_tt, aw_chunk_tt = compute_sampling_locations_and_attention_weights(
                    so_chunk,
                    aw_chunk,
                    ref_chunk,
                    spatial_shapes,
                    bs,
                    chunk_Q,
                    self.num_heads,
                    self.num_levels,
                    self.num_points,
                    self.device,
                    spatial_shapes_tt=spatial_shapes_tt,
                )
                ttnn.deallocate(so_chunk)
                ttnn.deallocate(aw_chunk)
                ttnn.deallocate(ref_chunk)

                chunk_out = multi_scale_deformable_attn_ttnn(
                    value_l_tts=value_l_tts,
                    sampling_locations_tt=loc_chunk_tt,
                    attention_weights_tt=aw_chunk_tt,
                    device=self.device,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                )
                ttnn.deallocate(loc_chunk_tt)
                ttnn.deallocate(aw_chunk_tt)
                output_chunks_device.append(chunk_out)

            ttnn.deallocate(sampling_offsets_flat)
            ttnn.deallocate(attention_weights_flat)
            for vl in value_l_tts:
                ttnn.deallocate(vl)

            output = ttnn.concat(output_chunks_device, dim=1)
            output = ttnn.linear(output, self.params["output_proj"]["weight"], bias=self.params["output_proj"]["bias"])
            output = ttnn.add(output, identity)
            for c in output_chunks_device:
                ttnn.deallocate(c)
            logger.info("  MSDeformAttn: done (trace_mode).")
            return output

        logger.info("  MSDeformAttn: computing sampling locations and attention weights on host...")
        so_torch = ttnn.to_torch(sampling_offsets_flat).float()
        aw_torch = ttnn.to_torch(attention_weights_flat).float()
        ttnn.deallocate(sampling_offsets_flat)
        ttnn.deallocate(attention_weights_flat)

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
            offset_normalizer = torch.stack([spatial_shapes[..., 1].float(), spatial_shapes[..., 0].float()], dim=-1)
            so_torch = so_torch / offset_normalizer[None, None, None, :, None, :]
            ref_xy = ref_pts.reshape(bs, num_queries, 1, ref_pts.shape[2], 1, 2)
            sampling_locations_torch = ref_xy + so_torch
        elif ref_pts.shape[-1] == 4:
            ref_xy = ref_pts[:, :, None, :, None, :2]
            ref_wh = ref_pts[:, :, None, :, None, 2:]
            sampling_locations_torch = ref_xy + (so_torch / self.num_points) * ref_wh * 0.5
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {ref_pts.shape[-1]}")

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

        value_l_tts = split_value_into_levels(value, spatial_shapes, self.num_heads, self.head_dim)

        num_upload_chunks = (num_queries + UPLOAD_CHUNK_QUERIES - 1) // UPLOAD_CHUNK_QUERIES
        logger.info(
            f"  MSDeformAttn: {num_upload_chunks} chunks (chunk={UPLOAD_CHUNK_QUERIES}), "
            f"grid+attention on device ({num_queries} queries)..."
        )
        output_chunks_host = []
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
                value_l_tts=value_l_tts,
                sampling_locations_tt=loc_chunk_tt,
                attention_weights_tt=aw_chunk_tt,
                device=self.device,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
            ttnn.deallocate(loc_chunk_tt)
            ttnn.deallocate(aw_chunk_tt)
            output_chunks_host.append(ttnn.to_torch(chunk_out))
            ttnn.deallocate(chunk_out)

        ttnn.deallocate(sampling_locations_tt)
        ttnn.deallocate(attention_weights_tt)
        for vl in value_l_tts:
            ttnn.deallocate(vl)

        output_cat = torch.cat(output_chunks_host, dim=1)
        output = ttnn.from_torch(
            output_cat,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        x = ttnn.linear(x, self.w1, bias=self.b1, activation="relu")
        x = ttnn.linear(x, self.w2, bias=self.b2)
        return ttnn.add(x, identity)


class TtDINOEncoderLayer:
    """Single encoder layer: MSDeformAttn -> LN -> FFN -> LN."""

    def __init__(self, params, device, embed_dims=256, num_heads=8, num_levels=5, num_points=4, trace_mode=False):
        self.self_attn = TtMSDeformAttn(
            params["self_attn"],
            device,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            trace_mode=trace_mode,
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
        spatial_shapes_tt=None,
        **kwargs,
    ):
        attn_kw = dict(
            query=query,
            value=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        if spatial_shapes_tt is not None:
            attn_kw["spatial_shapes_tt"] = spatial_shapes_tt
        query = self.self_attn(**attn_kw)
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

    def __init__(
        self,
        params,
        device,
        num_layers=6,
        embed_dims=256,
        num_heads=8,
        num_levels=5,
        num_points=4,
        profile_per_layer=False,
        trace_mode=True,
    ):
        self.device = device
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.profile_per_layer = profile_per_layer
        self.trace_mode = trace_mode
        self.layers = [
            TtDINOEncoderLayer(
                params["layers"][i],
                device,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                trace_mode=trace_mode,
            )
            for i in range(num_layers)
        ]

        self._ref_points_cache = None
        if trace_mode:
            from models.experimental.dino_5scale_swin_l.tt.tt_neck import DINO_NECK_LEVEL_SHAPES

            hw_list = [[DINO_NECK_LEVEL_SHAPES[l][2], DINO_NECK_LEVEL_SHAPES[l][3]] for l in range(num_levels)]
            spatial_shapes = torch.tensor(hw_list, dtype=torch.long)
            valid_ratios = torch.ones(1, num_levels, 2, dtype=torch.float32)
            valid_ratios_tt = ttnn.full(
                (1, num_levels, 2), 1.0, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self._ref_points_cache = TtDINOEncoder.get_encoder_reference_points(
                spatial_shapes, valid_ratios, device, valid_ratios_tt=valid_ratios_tt
            )
            ttnn.deallocate(valid_ratios_tt)

    @staticmethod
    def _linspace_ttnn(start, end, steps, device, dtype=ttnn.bfloat16):
        """Linspace on device: [start, ..., end] with steps elements. Steps must be >= 1."""
        if steps <= 1:
            return ttnn.full((1,), start, dtype=dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idx = ttnn.arange(0, steps, dtype=dtype, device=device)
        step_size = (end - start) / (steps - 1)
        return ttnn.add(ttnn.multiply(idx, step_size), start)

    @staticmethod
    def get_encoder_reference_points(spatial_shapes, valid_ratios, device, valid_ratios_tt=None):
        """
        Generate 2D reference points for encoder self-attention (pure TTNN on device).

        Returns: [B, sum(H_i*W_i), num_levels, 2] as ttnn.Tensor on device.
        When valid_ratios_tt is provided (trace_mode), use it instead of from_torch.
        """
        if hasattr(spatial_shapes, "tolist"):
            hw_list = spatial_shapes.tolist()
        else:
            hw_list = list(spatial_shapes)
        B = valid_ratios.shape[0] if valid_ratios is not None else valid_ratios_tt.shape[0]
        num_levels = len(hw_list)

        if valid_ratios_tt is not None:
            valid_ratios_tt = ttnn.to_memory_config(valid_ratios_tt, ttnn.L1_MEMORY_CONFIG)
        else:
            valid_ratios_tt = ttnn.from_torch(
                valid_ratios.float(),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        reference_points_list = []
        for lvl, hw in enumerate(hw_list):
            H, W = int(hw[0]), int(hw[1])
            if H * W == 0:
                continue

            ref_y_vals = TtDINOEncoder._linspace_ttnn(0.5, H - 0.5, H, device)
            ref_y_vals = ttnn.reshape(ref_y_vals, (H, 1))
            ref_y_vals = ttnn.repeat(ref_y_vals, (1, W))
            ref_y_vals = ttnn.reshape(ref_y_vals, (1, H * W))

            ref_x_vals = TtDINOEncoder._linspace_ttnn(0.5, W - 0.5, W, device)
            ref_x_vals = ttnn.reshape(ref_x_vals, (1, W))
            ref_x_vals = ttnn.repeat(ref_x_vals, (H, 1))
            ref_x_vals = ttnn.reshape(ref_x_vals, (1, H * W))

            vr_slice = ttnn.slice(valid_ratios_tt, [0, lvl, 0], [B, lvl + 1, 2], memory_config=ttnn.L1_MEMORY_CONFIG)
            vr_slice = ttnn.reshape(vr_slice, (B, 2))
            denom_y = ttnn.multiply(ttnn.reshape(vr_slice[:, 1:2], (B, 1)), float(H))
            denom_x = ttnn.multiply(ttnn.reshape(vr_slice[:, 0:1], (B, 1)), float(W))

            ref_y = ttnn.divide(ref_y_vals, denom_y)
            ref_x = ttnn.divide(ref_x_vals, denom_x)

            ref_level = ttnn.stack([ref_x, ref_y], dim=-1)
            reference_points_list.append(ref_level)
            ttnn.deallocate(ref_y_vals)
            ttnn.deallocate(ref_x_vals)

        reference_points = ttnn.concat(reference_points_list, dim=1)
        for t in reference_points_list:
            ttnn.deallocate(t)

        ref_bn12 = ttnn.reshape(reference_points, (B, reference_points.shape[1], 1, 2))
        valid_b1l2 = ttnn.reshape(valid_ratios_tt, (B, 1, num_levels, 2))
        reference_points = ttnn.multiply(ref_bn12, valid_b1l2)

        ttnn.deallocate(ref_bn12)
        ttnn.deallocate(valid_ratios_tt)

        return reference_points

    def __call__(
        self,
        feat,
        feat_pos,
        feat_mask,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        valid_ratios_tt=None,
        spatial_shapes_tt=None,
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
        if self.trace_mode and self._ref_points_cache is not None:
            B = feat.shape[0]
            reference_points = ttnn.repeat(self._ref_points_cache, (B, 1, 1, 1)) if B > 1 else self._ref_points_cache
        else:
            reference_points = self.get_encoder_reference_points(
                spatial_shapes, valid_ratios, device=self.device, valid_ratios_tt=valid_ratios_tt
            )
        logger.info(f"Reference points shape: {reference_points.shape}")

        query = feat
        for i, layer in enumerate(self.layers):
            logger.info(f"Encoder layer {i} starting...")
            layer_kw = dict(
                query=query,
                query_pos=feat_pos,
                key_padding_mask=feat_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
            )
            if spatial_shapes_tt is not None:
                layer_kw["spatial_shapes_tt"] = spatial_shapes_tt
            query = layer(**layer_kw)
            if not self.trace_mode:
                ttnn.synchronize_device(self.device)
            if self.profile_per_layer:
                ttnn.ReadDeviceProfiler(self.device)
            logger.info(f"Encoder layer {i} done.")
        return query
