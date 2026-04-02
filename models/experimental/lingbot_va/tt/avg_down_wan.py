# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager


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
        pc = self.parallel_config
        cm = self.ccl_manager
        dev = x.device()
        multi = pc is not None and cm is not None and dev is not None and dev.get_num_devices() > 1
        if multi:
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

        # Match prior PyTorch path: accumulate mean in float32, return bfloat16.
        x_work = ttnn.typecast(x_work, ttnn.float32)
        # Use x_work shapes so H/W match after height/width all_gather (local x may be sharded).
        B, T, H, W, C = (int(x_work.shape[i]) for i in range(5))
        x_bcthw = ttnn.permute(x_work, (0, 4, 1, 2, 3))

        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            pad_zeros = ttnn.zeros(
                (B, C, pad_t, H, W),
                device=x.device(),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            x_bcthw = ttnn.concat([pad_zeros, x_bcthw], dim=2)

        _b, c, tn, h, w = (int(x_bcthw.shape[i]) for i in range(5))
        ft, fs = self.factor_t, self.factor_s
        t1, h1, w1 = tn // ft, h // fs, w // fs

        x_bcthw = ttnn.reshape(x_bcthw, (_b, c, t1, ft, h1, fs, w1, fs))
        x_bcthw = ttnn.permute(x_bcthw, (0, 1, 3, 5, 7, 2, 4, 6))
        x_bcthw = ttnn.reshape(x_bcthw, (_b, c * self.factor, t1, h1, w1))
        x_bcthw = ttnn.reshape(x_bcthw, (_b, self.out_channels, self.group_size, t1, h1, w1))
        x_bcthw = ttnn.mean(x_bcthw, dim=2, keepdim=False)
        x_bthwc = ttnn.permute(x_bcthw, (0, 2, 3, 4, 1))
        tt_out = ttnn.typecast(x_bthwc, ttnn.bfloat16)

        if multi:
            # mesh_partition uses tilized slice rules on TILE tensors; arbitrary H/W (e.g. 160) then
            # fail TT_FATAL tile alignment. Match the old from_torch(..., ROW_MAJOR_LAYOUT) path.
            tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
            if pc.height_parallel.factor > 1:
                tt_out = ttnn.mesh_partition(tt_out, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
            if pc.width_parallel.factor > 1:
                tt_out = ttnn.mesh_partition(tt_out, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        return tt_out
