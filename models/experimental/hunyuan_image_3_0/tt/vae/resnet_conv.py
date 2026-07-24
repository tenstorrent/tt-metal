# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paired ResnetBlock 3x3 convs — shared spatial metadata and aligned H-chunk plans."""

from __future__ import annotations

import ttnn

from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d, conv3d_h_chunk_size
from models.tt_dit.layers.module import Module


class HunyuanResnetConvPair(Module):
    """ResnetBlock conv1 + conv2 with shared chunk geometry when shapes match.

    GroupNorm + SiLU sit between the two convs, so weights cannot be fused. When
    in==out (both convs are C->C at the same resolution), both convs share the
    same H-chunk strip height so slice/concat boundaries align across conv1 and
    conv2 on the spatial-sharded decode path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType,
        t: int,
        h: int,
        w: int,
    ) -> None:
        super().__init__()
        conv_kwargs = dict(mesh_device=mesh_device, dtype=dtype, t=t, h=h, w=w)
        self.conv1 = HunyuanSymmetricConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, **conv_kwargs
        )
        self.conv2 = HunyuanSymmetricConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, **conv_kwargs
        )
        self._same_channel_count = in_channels == out_channels

    def _apply_chunk_overrides(self, x_bthwc: ttnn.Tensor) -> None:
        _, t, h, w, _ = x_bthwc.shape
        valid_conv = self.conv1.spatial_sharded
        hc1 = conv3d_h_chunk_size(t=t, h=h, w=w, in_channels=self.conv1.in_channels, valid_conv=valid_conv)
        hc2 = conv3d_h_chunk_size(t=t, h=h, w=w, in_channels=self.conv2.in_channels, valid_conv=valid_conv)
        if self._same_channel_count and hc1 is not None and hc2 is not None:
            shared = max(hc1, hc2)
            self.conv1._h_chunk_override = shared
            self.conv2._h_chunk_override = shared
        else:
            self.conv1._h_chunk_override = hc1
            self.conv2._h_chunk_override = hc2

    def forward_conv1(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        self._apply_chunk_overrides(x_bthwc)
        return self.conv1(x_bthwc)

    def forward_conv2(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv2(x_bthwc)

    def enable_spatial(self, ccl, *, h_mesh_axis, w_mesh_axis) -> None:
        for conv in (self.conv1, self.conv2):
            conv.ccl = ccl
            conv.h_mesh_axis = h_mesh_axis
            conv.w_mesh_axis = w_mesh_axis
            conv.spatial_sharded = True

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        raise RuntimeError("HunyuanResnetConvPair: use forward_conv1/forward_conv2 from ResnetBlockTTNN")
