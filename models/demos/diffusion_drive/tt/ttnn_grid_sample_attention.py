# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN port of GridSampleCrossBEVAttention — Stage-5 de-risk.

The README previously listed ``F.grid_sample`` as the key Stage-5 blocker with
"no TTNN equivalent yet".  That is no longer true: this build ships
``ttnn.grid_sample`` (mode="bilinear", padding_mode="zeros", align_corners=False
— exactly what the reference module uses).  This wrapper runs the deformable
sampling on-device and matches the PyTorch reference to PCC ≳ 0.99.

Scope: the actual blocker (`grid_sample`) runs on TTNN here.  The three small
weight layers around it (`attention_weights` Linear, `value_proj` 3×3 Conv+ReLU,
`output_proj` Linear) are kept as the reference torch submodules so the block
stays weight-faithful; they are trivially portable later (standard TTNN
linear/conv ops) and are not the thing that blocked Stage 5.

Reference: ``reference.model.GridSampleCrossBEVAttention``.
"""

from __future__ import annotations

import torch

import ttnn


class TtnnGridSampleCrossBEVAttention:
    """Drop-in for GridSampleCrossBEVAttention.forward with grid_sample on device.

    Parameters
    ----------
    ref : GridSampleCrossBEVAttention
        Pre-loaded, eval-mode reference module (provides the weight layers and
        ``config.lidar_max_x`` / ``lidar_max_y``).
    device : ttnn.Device
    """

    def __init__(self, ref, device: ttnn.Device) -> None:
        self._ref = ref
        self._device = device

    def _grid_sample(self, value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """ttnn.grid_sample bridge.

        Args:
            value: (B, C, H, W) float32 — BEV value map.
            grid:  (B, K, T, 2) float32 — normalised [-1, 1] sample coords.
        Returns:
            (B, C, K, T) float32 — sampled features (matches F.grid_sample layout).
        """
        B, C, H, W = value.shape
        _, K, T, _ = grid.shape
        v_nhwc = value.permute(0, 2, 3, 1).contiguous()
        v_tt = ttnn.from_torch(v_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        g_tt = ttnn.from_torch(
            grid.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device
        )
        out_tt = ttnn.grid_sample(v_tt, g_tt)  # (B, K, T, C) NHWC
        out = ttnn.to_torch(out_tt).reshape(B, K, T, C).permute(0, 3, 1, 2).float()  # (B, C, K, T)
        return out

    def __call__(
        self,
        queries: torch.Tensor,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        spatial_shape,
    ) -> torch.Tensor:
        ref = self._ref
        bs, num_queries, num_points, _ = traj_points.shape

        # Normalise to [-1, 1] and swap x↔y (identical to the reference).
        norm = traj_points.clone()
        norm[..., 0] = norm[..., 0] / ref.config.lidar_max_y
        norm[..., 1] = norm[..., 1] / ref.config.lidar_max_x
        norm = norm[..., [1, 0]]

        attn_w = ref.attention_weights(queries).view(bs, num_queries, num_points).softmax(-1)
        value = ref.value_proj(bev_feature)  # B, 256, H, W

        grid = norm.view(bs, num_queries, num_points, 2)
        sampled = self._grid_sample(value, grid)  # B, 256, K, T  (grid_sample on TTNN)

        attn_w = attn_w.unsqueeze(1)  # B, 1, K, T
        out = (attn_w * sampled).sum(dim=-1)  # B, 256, K
        out = out.permute(0, 2, 1).contiguous()  # B, K, 256
        out = ref.output_proj(out)
        return ref.dropout(out) + queries
