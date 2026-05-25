# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Scale Deformable Attention (MSDeformAttn) in ttnn.

ED-Pose config: d_model=256, n_heads=8, n_levels=5, n_points=4, d_per_head=32.

Two execution paths:
  - Device path (encoder): offsets stay on device, sampling grids computed via
    device-side arithmetic, grid_sample with use_precomputed_grid=False.
    Eliminates host round-trips (to_torch/from_torch per layer).
  - Legacy CPU path (decoder): offsets transferred to host for 6D reshape,
    bilinear grid precomputed on CPU, grid_sample with use_precomputed_grid=True.
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
        Vectorized bilinear grid precomputation for CPU path.
        Reproduces C++ bit-packing: h0/w0 as int16 bit-cast to bfloat16.
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

    def _fused_grid_for_level(self, offsets_level, ref_level, norm_level, H, W, N_M, Lq, P):
        """Fused sampling_locs + bilinear precompute for one level.

        Avoids creating the full (N, Lq, M, L, P, 2) intermediate tensor
        by computing coordinates and bilinear weights per level (~20MB vs 103MB).
        """
        locs = ref_level + offsets_level / norm_level
        grid_l = locs.mul_(2).sub_(1)
        grid_l = grid_l.transpose(1, 2).flatten(0, 1).contiguous().float()

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

    def _compute_attn_weights(self, attn_w_tt, M, L, P):
        """Reshape and softmax attention weights on device."""
        old = attn_w_tt
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        s = list(attn_w_tt.shape)
        N, Lq = s[0], s[1]
        attn_w_tt = ttnn.reshape(attn_w_tt, (1, 1, N * Lq * M, L * P))
        old = attn_w_tt
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        old = attn_w_tt
        attn_w_tt = ttnn.softmax(attn_w_tt, dim=-1)
        ttnn.deallocate(old)
        old = attn_w_tt
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        attn_w_tt = ttnn.reshape(attn_w_tt, (N, Lq, M, L * P))
        attn_w_tt = ttnn.transpose(attn_w_tt, 1, 2)
        attn_w_tt = ttnn.reshape(attn_w_tt, (N * M, Lq, L * P, 1))
        return attn_w_tt, N, Lq

    def _sample_and_aggregate(self, value_tt, attn_w_tt, spatial_shapes, input_level_start_index,
                              N, M, D, L, P, Lq, Len_in, grid_fn):
        """Per-level grid_sample + attention-weighted aggregation.

        Defers sum(dim=2) until after all levels are accumulated to reduce
        the number of device reduce ops from L to 1.
        """
        result_accum = None
        for lid, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            start = int(input_level_start_index[lid])
            end = start + H * W

            val = ttnn.slice(value_tt, [0, start, 0, 0], [N, end, M, D])
            val = ttnn.transpose(val, 1, 2)
            val = ttnn.reshape(val, (N * M, H, W, D))

            grid_tt, use_precomputed = grid_fn(lid, H, W, N * M, Lq, P)

            sampled = ttnn.grid_sample(
                val, grid_tt, use_precomputed_grid=use_precomputed, align_corners=False)
            ttnn.deallocate(val)
            ttnn.deallocate(grid_tt)

            attn_l_tt = ttnn.slice(
                attn_w_tt, [0, 0, lid * P, 0], [N * M, Lq, (lid + 1) * P, 1])
            weighted = ttnn.multiply(attn_l_tt, sampled)
            ttnn.deallocate(attn_l_tt)
            ttnn.deallocate(sampled)

            if result_accum is None:
                result_accum = weighted
            else:
                old = result_accum
                result_accum = ttnn.add(result_accum, weighted)
                ttnn.deallocate(old)
                ttnn.deallocate(weighted)

        old = result_accum
        result_accum = ttnn.sum(result_accum, dim=2)
        ttnn.deallocate(old)
        return result_accum

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
        ref_levels_tt=None,
        inv_norm_levels_tt=None,
        cpu_ref_levels=None,
        cpu_norm_levels=None,
    ):
        """
        Args:
            query: ttnn (N, Lq, d_model) or torch
            reference_points: torch (N, Lq, n_levels, 2|4)
            input_flatten: ttnn (N, Len_in, d_model) or torch
            input_spatial_shapes: torch (n_levels, 2)
            input_level_start_index: torch (n_levels,)
            input_padding_mask: optional torch (N, Len_in) bool
            normalizer: optional precomputed torch (n_levels, 2) float
            ref_expanded: optional precomputed torch (N, Lq, 1, n_levels, 1, 2) float
            ref_levels_tt: optional list of ttnn (1, Lq, 1, 2) per level — device path
            inv_norm_levels_tt: optional list of ttnn (1, 1, 1, 2) per level — device path
        Returns:
            ttnn (N, Lq, d_model)
        """
        query_tt = self._ensure_tt(query)
        query_is_temp = not isinstance(query, ttnn.Tensor)
        input_tt = self._ensure_tt(input_flatten)

        M = self.n_heads
        D = self.d_per_head
        L = self.n_levels
        P = self.n_points

        value_tt = ttnn.linear(input_tt, self.value_proj_w, bias=self.value_proj_b)
        offsets_tt = ttnn.linear(query_tt, self.sampling_offsets_w, bias=self.sampling_offsets_b)
        attn_w_tt = ttnn.linear(query_tt, self.attention_weights_w, bias=self.attention_weights_b)
        if query_is_temp:
            ttnn.deallocate(query_tt)

        spatial_shapes = input_spatial_shapes
        use_device_grid = ref_levels_tt is not None and inv_norm_levels_tt is not None

        if not use_device_grid:
            offsets_t = ttnn.to_torch(offsets_tt).float()
            ttnn.deallocate(offsets_tt)

        if input_padding_mask is not None:
            inv_mask = (~input_padding_mask).float().unsqueeze(-1).to(torch.bfloat16)
            inv_mask_tt = ttnn.from_torch(
                inv_mask, layout=ttnn.TILE_LAYOUT,
                device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            old = value_tt
            value_tt = ttnn.multiply(value_tt, inv_mask_tt)
            ttnn.deallocate(old)
            ttnn.deallocate(inv_mask_tt)

        attn_w_tt, N, Lq = self._compute_attn_weights(attn_w_tt, M, L, P)

        old = value_tt
        value_tt = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        Len_in = list(value_tt.shape)[1]
        value_tt = ttnn.reshape(value_tt, (N, Len_in, M, D))

        if use_device_grid:
            # === DEVICE PATH: compute sampling grids entirely on device ===
            old = offsets_tt
            offsets_tt = ttnn.to_layout(offsets_tt, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(old)
            offsets_tt = ttnn.reshape(offsets_tt, (N, Lq, M, L * P * 2))
            offsets_tt = ttnn.transpose(offsets_tt, 1, 2)
            offsets_tt = ttnn.reshape(offsets_tt, (N * M, Lq, L, P * 2))

            def device_grid_fn(lid, H, W, NM, Lq_, P_):
                off_l = ttnn.slice(offsets_tt, [0, 0, lid, 0], [NM, Lq_, lid + 1, P_ * 2])
                off_l = ttnn.reshape(off_l, (NM, Lq_, P_, 2))
                old_off = off_l
                off_l = ttnn.multiply(off_l, inv_norm_levels_tt[lid])
                ttnn.deallocate(old_off)
                locs_l = ttnn.add(off_l, ref_levels_tt[lid])
                ttnn.deallocate(off_l)
                old_grid = ttnn.multiply(locs_l, 2.0)
                ttnn.deallocate(locs_l)
                grid_l = ttnn.subtract(old_grid, 1.0)
                ttnn.deallocate(old_grid)
                return grid_l, False

            result_accum = self._sample_and_aggregate(
                value_tt, attn_w_tt, spatial_shapes, input_level_start_index,
                N, M, D, L, P, Lq, Len_in, device_grid_fn)
            ttnn.deallocate(offsets_tt)

        else:
            # === CPU PATH: offsets already transferred above for overlap ===
            offsets_t = offsets_t.view(N, Lq, M, L, P, 2)

            if reference_points.shape[-1] == 2:
                if cpu_ref_levels is None:
                    if normalizer is None:
                        normalizer = torch.stack(
                            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
                        ).float()
                    ref_f = reference_points.float()
                    cpu_ref_levels = [ref_f[:, :, lid, :].unsqueeze(2).unsqueeze(3) for lid in range(L)]
                    cpu_norm_levels = [normalizer[lid].view(1, 1, 1, 1, 2) for lid in range(L)]

                def cpu_grid_fn(lid, H, W, NM, Lq_, P_):
                    off_l = offsets_t[:, :, :, lid, :, :]
                    precomputed_tt = self._fused_grid_for_level(
                        off_l, cpu_ref_levels[lid], cpu_norm_levels[lid], H, W, NM, Lq_, P_)
                    return precomputed_tt, True

            elif reference_points.shape[-1] == 4:
                reference_points_f = reference_points.float()
                sampling_locs = (
                    reference_points_f[:, :, None, :, None, :2]
                    + offsets_t / P * reference_points_f[:, :, None, :, None, 2:] * 0.5
                )
                sampling_grids = sampling_locs.mul_(2).sub_(1)

                def cpu_grid_fn(lid, H, W, NM, Lq_, P_):
                    grid_l = (
                        sampling_grids[:, :, :, lid]
                        .transpose(1, 2)
                        .flatten(0, 1)
                        .contiguous()
                        .float()
                    )
                    precomputed_tt = self._precompute_bilinear_grid_python(grid_l, H, W, NM, Lq_, P_)
                    return precomputed_tt, True
            else:
                raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

            result_accum = self._sample_and_aggregate(
                value_tt, attn_w_tt, spatial_shapes, input_level_start_index,
                N, M, D, L, P, Lq, Len_in, cpu_grid_fn)

        ttnn.deallocate(value_tt)
        ttnn.deallocate(attn_w_tt)

        old = result_accum
        result = ttnn.to_layout(result_accum, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        result = ttnn.reshape(result, (N, M, Lq, D))
        result = ttnn.transpose(result, 1, 2)
        result = ttnn.reshape(result, (N, Lq, M * D))
        old = result
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)

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
