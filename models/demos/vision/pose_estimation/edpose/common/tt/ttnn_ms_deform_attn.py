# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Scale Deformable Attention (MSDeformAttn) in ttnn.

ED-Pose config: d_model=256, n_heads=8, n_levels=5, n_points=4, d_per_head=32.

Value stays on device throughout. Per-level value preparation (slice, transpose,
reshape to NHWC) and weighted aggregation (concat + matmul) run on device.
Grid coordinates are computed on host (6D reference_points logic) and transferred
with precomputed bilinear weights + K=4 batching for efficient grid_sample.
Offsets/attention_weights go to host for 6D reshape and softmax (small tensors).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTMSDeformAttn(LightweightModule):

    def __init__(self, device, state_dict, prefix, d_model=256, n_heads=8, n_levels=5, n_points=4):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_per_head = d_model // n_heads

        p = f"{prefix}." if prefix else ""
        self.value_proj_w = self._weight(state_dict[f"{p}value_proj.weight"])
        self.value_proj_b = self._bias(state_dict[f"{p}value_proj.bias"])

        self.sampling_offsets_w = self._weight(state_dict[f"{p}sampling_offsets.weight"])
        self.sampling_offsets_b = self._bias(state_dict[f"{p}sampling_offsets.bias"])

        self.attention_weights_w = self._weight(state_dict[f"{p}attention_weights.weight"])
        self.attention_weights_b = self._bias(state_dict[f"{p}attention_weights.bias"])

        self.output_proj_w = self._weight(state_dict[f"{p}output_proj.weight"])
        self.output_proj_b = self._bias(state_dict[f"{p}output_proj.bias"])

    def _weight(self, w):
        return ttnn.from_torch(
            w.T.contiguous().to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bias(self, b):
        return ttnn.from_torch(
            b.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _precompute_bilinear_grid_python(self, grid_l, H, W, N_M, Lq, P):
        """
        Vectorized replacement for ttnn.prepare_grid_sample_grid (bilinear, align_corners=False, zeros padding).

        Reproduces the C++ bit-packing: h0/w0 stored as int16 bit-cast to bfloat16,
        weights stored as normal bfloat16.

        Args:
            grid_l: (N*M, Lq, P, 2) float32 in [-1,1], last dim = (x, y)
            H, W: feature map spatial dims
            N_M, Lq, P: batch*heads, queries, points
        Returns:
            ttnn tensor (N*M, Lq, 1, 6*P) bfloat16, ROW_MAJOR on device
        """
        x = grid_l[..., 0]
        y = grid_l[..., 1]

        h_coord = (y + 1.0) * (H * 0.5) - 0.5
        w_coord = (x + 1.0) * (W * 0.5) - 0.5

        h0 = torch.floor(h_coord)
        w0 = torch.floor(w_coord)

        h_frac = h_coord - h0
        w_frac = w_coord - w0

        h0i = h0.to(torch.int32)
        w0i = w0.to(torch.int32)

        h0v = (h0i >= 0) & (h0i < H)
        h1v = ((h0i + 1) >= 0) & ((h0i + 1) < H)
        w0v = (w0i >= 0) & (w0i < W)
        w1v = ((w0i + 1) >= 0) & ((w0i + 1) < W)

        hfi = 1.0 - h_frac
        wfi = 1.0 - w_frac

        wt_nw = (hfi * wfi * (h0v & w0v).float()).to(torch.bfloat16)
        wt_ne = (hfi * w_frac * (h0v & w1v).float()).to(torch.bfloat16)
        wt_sw = (h_frac * wfi * (h1v & w0v).float()).to(torch.bfloat16)
        wt_se = (h_frac * w_frac * (h1v & w1v).float()).to(torch.bfloat16)

        h0_bf16 = h0.clamp(-32768, 32767).to(torch.int16).view(torch.bfloat16)
        w0_bf16 = w0.clamp(-32768, 32767).to(torch.int16).view(torch.bfloat16)

        packed = torch.stack([h0_bf16, w0_bf16, wt_nw, wt_ne, wt_sw, wt_se], dim=-1)
        packed = packed.reshape(N_M, Lq, 1, 6 * P)

        return ttnn.from_torch(packed.contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        normalizer=None,
        ref_expanded=None,
    ):
        """
        Args:
            query: ttnn (N, Lq, d_model) TILE_LAYOUT or torch (N, Lq, d_model)
            reference_points: torch (N, Lq, n_levels, 2|4) in [0,1]
            input_flatten: ttnn (N, Len_in, d_model) TILE_LAYOUT or torch
            input_spatial_shapes: torch (n_levels, 2) as int — [(H0,W0),…]
            input_level_start_index: torch (n_levels,) as int
            input_padding_mask: optional torch (N, Len_in) bool
        Returns:
            ttnn (N, Lq, d_model) TILE_LAYOUT
        """
        query_tt = self._ensure_tt(query)
        input_tt = self._ensure_tt(input_flatten)

        M = self.n_heads
        D = self.d_per_head
        L = self.n_levels
        P = self.n_points

        value_tt = ttnn.linear(input_tt, self.value_proj_w, bias=self.value_proj_b)
        offsets_tt = ttnn.linear(query_tt, self.sampling_offsets_w, bias=self.sampling_offsets_b)
        attn_w_tt = ttnn.linear(query_tt, self.attention_weights_w, bias=self.attention_weights_b)

        if input_padding_mask is not None:
            inv_mask = (~input_padding_mask).float().unsqueeze(-1).to(torch.bfloat16)
            inv_mask_tt = ttnn.from_torch(
                inv_mask, layout=ttnn.TILE_LAYOUT,
                device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            value_tt = ttnn.multiply(value_tt, inv_mask_tt)
            ttnn.deallocate(inv_mask_tt)

        offsets_t = ttnn.to_torch(offsets_tt).float()
        ttnn.deallocate(offsets_tt)

        N = offsets_t.shape[0]
        Lq = offsets_t.shape[1]
        Len_in = sum(int(H) * int(W) for H, W in input_spatial_shapes)

        # Device-side softmax: keep attn_w on device
        # attn_w_tt: (N, Lq, M*L*P) TILE → softmax over groups of L*P per head
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        attn_w_tt = ttnn.reshape(attn_w_tt, (1, 1, N * Lq * M, L * P))
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.TILE_LAYOUT)
        attn_w_tt = ttnn.softmax(attn_w_tt, dim=-1)
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        attn_w_tt = ttnn.reshape(attn_w_tt, (N, Lq, M, L * P))
        attn_w_tt = ttnn.transpose(attn_w_tt, 1, 2)
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.TILE_LAYOUT)
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        attn_w_tt = ttnn.reshape(attn_w_tt, (N * M, Lq, L * P, 1))

        value_tt = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
        value_tt = ttnn.reshape(value_tt, (N, Len_in, M, D))

        offsets_t = offsets_t.view(N, Lq, M, L, P, 2)

        spatial_shapes = input_spatial_shapes
        if reference_points.shape[-1] == 2:
            if normalizer is None:
                normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
                ).float()
            if ref_expanded is None:
                ref_expanded = reference_points.float()[:, :, None, :, None, :]
            sampling_locs = ref_expanded + offsets_t / normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            reference_points_f = reference_points.float()
            sampling_locs = (
                reference_points_f[:, :, None, :, None, :2]
                + offsets_t / P * reference_points_f[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        sampling_grids = sampling_locs.mul_(2).sub_(1)

        result_accum = None
        for lid, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            start = int(input_level_start_index[lid])
            end = start + H * W

            val = ttnn.slice(value_tt, [0, start, 0, 0], [N, end, M, D])
            val = ttnn.transpose(val, 1, 2)
            val = ttnn.to_layout(val, ttnn.TILE_LAYOUT)
            val = ttnn.to_layout(val, ttnn.ROW_MAJOR_LAYOUT)
            val = ttnn.reshape(val, (N * M, H, W, D))

            grid_l = (
                sampling_grids[:, :, :, lid]
                .transpose(1, 2)
                .flatten(0, 1)
                .contiguous()
                .float()
            )
            precomputed_tt = self._precompute_bilinear_grid_python(grid_l, H, W, N * M, Lq, P)

            sampled = ttnn.grid_sample(
                val, precomputed_tt,
                use_precomputed_grid=True, align_corners=False,
            )
            ttnn.deallocate(val)
            ttnn.deallocate(precomputed_tt)

            attn_l_tt = ttnn.slice(
                attn_w_tt, [0, 0, lid * P, 0], [N * M, Lq, (lid + 1) * P, 1]
            )
            weighted = ttnn.multiply(attn_l_tt, sampled)
            ttnn.deallocate(attn_l_tt)
            ttnn.deallocate(sampled)
            level_sum = ttnn.sum(weighted, dim=2)
            ttnn.deallocate(weighted)

            if result_accum is None:
                result_accum = level_sum
            else:
                result_accum = ttnn.add(result_accum, level_sum)
                ttnn.deallocate(level_sum)

        ttnn.deallocate(value_tt)
        ttnn.deallocate(attn_w_tt)

        result = result_accum
        result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
        result = ttnn.reshape(result, (N, M, Lq, D))
        result = ttnn.transpose(result, 1, 2)
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
        result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
        result = ttnn.reshape(result, (N, Lq, M * D))
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)

        output = ttnn.linear(result, self.output_proj_w, bias=self.output_proj_b)
        ttnn.deallocate(result)
        return output

    def _ensure_tt(self, x):
        if isinstance(x, torch.Tensor):
            return ttnn.from_torch(
                x.to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return x
