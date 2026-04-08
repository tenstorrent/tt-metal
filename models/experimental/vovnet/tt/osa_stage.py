# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.vovnet.tt.osa_block import TtOsaBlock
import ttnn

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


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
        if use_signpost:
            signpost(header="osa_stage")

        if self.pool:
            N, C, H, W = x.shape
            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.reshape(x, (1, 1, N * H * W, C))

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
                applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            )
            x = ttnn.reshape(x, (N, H // 2, W // 2, C))
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.permute(x, (0, 3, 1, 2))

        for i, module in enumerate(self.blocks):
            x = module.forward(x)
        return x
