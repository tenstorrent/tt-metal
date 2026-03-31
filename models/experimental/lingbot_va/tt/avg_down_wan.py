# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager


def _bthwc_to_torch_cpu(x: ttnn.Tensor) -> torch.Tensor:
    """Host tensor for PyTorch pooling; mesh tensors need a single-device readback."""
    dev = x.device()
    if dev is not None and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
    return ttnn.to_torch(x).float()


class TtAvgDown3D(Module):
    """
    Spatial-temporal downsampling via averaging.

    Takes [B, T, H, W, C_in] and produces
    [B, T/factor_t, H/factor_s, W/factor_s, C_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
        *,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.factor = factor_t * factor_s * factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        B, T, H, W, C = x.shape

        pc = self.parallel_config
        cm = self.ccl_manager
        dev = x.device()
        multi = pc is not None and cm is not None and dev is not None and dev.get_num_devices() > 1
        if multi:
            # Fractured B,T,H,W,C must be gathered before host pooling (same axes as WanAttentionBlock).
            x_work = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            if pc.height_parallel.factor > 1:
                x_work = cm.all_gather_persistent_buffer(
                    x_work, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=True
                )
            if pc.width_parallel.factor > 1:
                x_work = cm.all_gather_persistent_buffer(
                    x_work, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=True
                )
            x_work = ttnn.to_layout(x_work, ttnn.ROW_MAJOR_LAYOUT)
        else:
            x_work = x

        # Keep float32 pooling parity with the PyTorch reference implementation.
        x_torch = _bthwc_to_torch_cpu(x_work)
        x_bcthw = x_torch.permute(0, 4, 1, 2, 3).contiguous()
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            x_bcthw = torch.nn.functional.pad(x_bcthw, (0, 0, 0, 0, pad_t, 0))

        B, C, T, H, W = x_bcthw.shape

        x_bcthw = x_bcthw.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )

        x_bcthw = x_bcthw.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x_bcthw = x_bcthw.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )

        x_bcthw = x_bcthw.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )

        x_bcthw = x_bcthw.mean(dim=2)
        x_bthwc = x_bcthw.permute(0, 2, 3, 4, 1).to(torch.bfloat16).contiguous()

        tt_out = ttnn.from_torch(
            x_bthwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=x.device(),
        )
        if multi:
            if pc.height_parallel.factor > 1:
                tt_out = ttnn.mesh_partition(tt_out, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
            if pc.width_parallel.factor > 1:
                tt_out = ttnn.mesh_partition(tt_out, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        return tt_out
