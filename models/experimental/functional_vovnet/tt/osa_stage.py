# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from models.experimental.functional_vovnet.tt.osa_block import TtOsaBlock
import ttnn


class TtOsaStage:
    def __init__(
        self,
        block_per_stage: int = 1,
        downsample=True,
        base_address=None,
        device=None,
        state_dict=None,
        parameters=None,
    ):
        self.device = device
        self.base_address = f"{base_address}.blocks.0"
        self.state_dict = state_dict
        self.maxpool_pad = 0
        self.maxpool_stride = 2
        self.maxpool_kernel = 3
        self.maxpool_dilation = 1
        self.cores_x = device.core_grid.x
        self.cores_y = device.core_grid.y
        self.max_cores = self.cores_x * self.cores_y
        if downsample:
            self.pool = True
        else:
            self.pool = False

        self.blocks = []
        for i in range(block_per_stage):
            self.blocks += [
                TtOsaBlock(
                    base_address=self.base_address,
                    parameters=parameters,
                    device=self.device,
                )
            ]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.pool:
            N, C, H, W = x.shape
            out_h = (
                int(((H + 2 * self.maxpool_pad - (self.maxpool_dilation * 3 - 1) - 1) / self.maxpool_stride) + 1) + 1
            )
            out_w = (
                int(((W + 2 * self.maxpool_pad - (self.maxpool_dilation * 3 - 1) - 1) / self.maxpool_stride) + 1) + 1
            )
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.reshape(x, (1, 1, N * H * W, C))
            parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                batch_size=N,
                input_channels=C,
                output_height=out_h,
                output_width=out_w,
                output_channels=C,
                compute_grid_size=self.device.compute_with_storage_grid_size(),
                block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                enable_channels_padding=False,
                is_out_tiled=False,
            )
            sharded_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
                tensor_shape=x.shape,
                parallel_config=parallel_config,
                tile_size=1,
            )

            x = ttnn.to_memory_config(x, sharded_memory_config)
            x = ttnn.max_pool2d(
                input_tensor=x,
                batch_size=N,
                input_h=H,
                input_w=W,
                channels=C,
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[0, 0],
                dilation=[1, 1],
                ceil_mode=True,
            )
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.reshape(x, (N, H // 2, W // 2, C))

            x = ttnn.permute(x, (0, 3, 1, 2))

        for i, module in enumerate(self.blocks):
            x = module.forward(x)
        return x
