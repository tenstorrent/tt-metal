# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN port of GridSampleCrossBEVAttention — fully on-device.

The README previously listed ``F.grid_sample`` as the key blocker with "no TTNN
equivalent yet".  That is no longer true: this build ships ``ttnn.grid_sample``
(mode="bilinear", padding_mode="zeros", align_corners=False — exactly what the
reference module uses).

This wrapper now runs the *entire* deformable-attention block on-device:

  * ``value_proj``         3×3 Conv2d(256→256)+ReLU   → ttnn.conv2d + ttnn.relu
  * ``ttnn.grid_sample``   bilinear deformable sample  → on device
  * ``attention_weights``  Linear(256→T)+softmax       → ttnn.linear + ttnn.softmax
  * weighted aggregation   Σ_t w·sample (the attention) → batched ttnn.matmul
  * ``output_proj``        Linear(256→256)             → ttnn.linear
  * residual add                                       → ttnn.add

The only host-side step left is normalising the trajectory waypoints into the
[-1, 1] sampling grid (two scalar divides + an x↔y swap) — that is coordinate
prep for the ``grid_sample`` API boundary, not a model compute op.

Matches the PyTorch reference to PCC ≳ 0.99.

Reference: ``reference.model.GridSampleCrossBEVAttention``.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.diffusion_drive.tt.ttnn_perception import _prep_linear
from models.demos.diffusion_drive.tt.ttnn_resnet34 import _RELU_ACT, _ttnn_conv2d, prep_conv_weights


class TtnnGridSampleCrossBEVAttention:
    """Drop-in for GridSampleCrossBEVAttention.forward, all compute on device.

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

        # value_proj = Sequential(Conv2d(256, 256, 3, padding=1, bias=True), ReLU)
        vconv = ref.value_proj[0]
        self._vp_cin = vconv.in_channels
        self._vp_cout = vconv.out_channels
        self._vp_w, self._vp_b = prep_conv_weights(
            vconv.weight.detach().to(torch.bfloat16), vconv.bias.detach().to(torch.bfloat16)
        )

        # attention_weights Linear(embed_dims → num_points) and output_proj Linear
        self._aw_w, self._aw_b = _prep_linear(ref.attention_weights, device)
        self._op_w, self._op_b = _prep_linear(ref.output_proj, device)

    def _value_proj(self, bev_feature: torch.Tensor):
        """value_proj conv+relu on device → ROW_MAJOR (B,H,W,256) ready for grid_sample."""
        B, C, H, W = bev_feature.shape
        x_nhwc = bev_feature.permute(0, 2, 3, 1).contiguous().reshape(1, 1, B * H * W, C).to(torch.bfloat16)
        xt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._device)
        # Reassign self._vp_w/_b to the device-prepared weights so each per-step
        # value_proj reuses them (trace-safe; no host re-upload of the conv weights).
        out, Ho, Wo, self._vp_w, self._vp_b = _ttnn_conv2d(
            self._device, xt, self._vp_w, self._vp_b, B, H, W, C, self._vp_cout, 3, 1, 1, activation=_RELU_ACT
        )
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (B, Ho, Wo, self._vp_cout))

    def _dev(self, q_tt, traj_points: torch.Tensor, bev_feature: torch.Tensor, spatial_shape):
        """Device-in/device-out core: ``q_tt`` is a device (B,K,D) tile tensor and the
        result stays on device (no to_torch).  ``traj_points``/``bev_feature`` are the
        per-step torch loop inputs (grid coords + BEV map); only the grid normalisation
        and the value_proj input crossing are on host, as in ``__call__``."""
        ref = self._ref
        bs, num_queries, num_points, _ = traj_points.shape

        # host: normalise waypoints into the [-1,1] sampling grid (API prep)
        norm = traj_points.clone()
        norm[..., 0] = norm[..., 0] / ref.config.lidar_max_y
        norm[..., 1] = norm[..., 1] / ref.config.lidar_max_x
        norm = norm[..., [1, 0]]  # swap x↔y
        grid = norm.view(bs, num_queries, num_points, 2).contiguous()

        value_nhwc = self._value_proj(bev_feature)  # ROW_MAJOR (B,H,W,256)
        g_tt = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        sampled = ttnn.grid_sample(value_nhwc, g_tt)  # (B,K,T,256) ROW_MAJOR

        attn_w = ttnn.linear(q_tt, self._aw_w, bias=self._aw_b)  # (B,K,T)
        attn_w = ttnn.softmax(attn_w, dim=-1)

        # weighted aggregation out[b,k,:] = Σ_t w[b,k,t]·sampled[b,k,t,:] via batched matmul
        sampled = ttnn.to_layout(ttnn.reshape(sampled, (bs * num_queries, num_points, self._vp_cout)), ttnn.TILE_LAYOUT)
        aw = ttnn.reshape(attn_w, (bs * num_queries, 1, num_points))
        out = ttnn.matmul(aw, sampled)  # (B*K, 1, 256)
        out = ttnn.reshape(out, (bs, num_queries, self._vp_cout))

        # output_proj + residual (dropout is identity at eval)
        out = ttnn.linear(out, self._op_w, bias=self._op_b)
        return ttnn.add(out, q_tt)

    def __call__(
        self,
        queries: torch.Tensor,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        spatial_shape,
    ) -> torch.Tensor:
        bs, num_queries, D = queries.shape
        q_tt = ttnn.from_torch(queries.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._device)
        out = self._dev(q_tt, traj_points, bev_feature, spatial_shape)
        return ttnn.to_torch(out).reshape(bs, num_queries, D).float()
