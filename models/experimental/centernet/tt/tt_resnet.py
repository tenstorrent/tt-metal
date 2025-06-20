# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.centernet.tt.common import TtConv


class TtBasicBlock:
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        device=None,
        parameters=None,
        base_address=None,
    ):
        self.device = device
        self.parameters = parameters
        self.base_address = base_address
        self.conv1 = TtConv(
            device=device,
            parameters=parameters,
            path=f"{base_address}.conv1",
            conv_params=[stride, stride, 1, 1],
            fused_op=True,
            activation="relu",
        )

        self.conv2 = TtConv(
            device=device, parameters=parameters, path=f"{base_address}.conv2", conv_params=[1, 1, 1, 1], fused_op=True
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = ttnn.to_layout(out, layout=ttnn.TILE_LAYOUT)
        identity = ttnn.to_layout(identity, layout=ttnn.TILE_LAYOUT)
        # out += identity
        out = ttnn.add(out, identity, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.relu(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        return out


class TtResNet:
    def __init__(
        self,
        block,
        layers,
        groups: int = 1,
        width_per_group: int = 64,
        parameters=None,
        base_address=None,
        replace_stride_with_dilation=None,
        norm_layer=None,
        device=None,
    ) -> None:
        self.parameters = parameters
        self.device = device
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.device = device
        replace_stride_with_dilation = [False, False, False]

        self.conv1 = TtConv(
            device=device,
            parameters=parameters,
            path=f"{base_address}.conv1",
            conv_params=[2, 2, 3, 3],
            fused_op=True,
            activation="relu",
        )

        self.layer1 = self._make_layer(
            block, 64, layers[0], parameters=parameters, base_address=f"{base_address}.layer1", device=device
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            parameters=parameters,
            base_address=f"{base_address}.layer2",
            device=device,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            parameters=parameters,
            base_address=f"{base_address}.layer3",
            device=device,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            parameters=parameters,
            base_address=f"{base_address}.layer4",
            device=device,
        )

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        device=None,
        parameters=None,
        base_address=None,
    ):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes:
            downsample = TtConv(
                device=device,
                parameters=parameters,
                path=f"{base_address}.0.downsample.0",
                conv_params=[2, 2, 0, 0],
                fused_op=True,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                parameters=parameters,
                base_address=f"{base_address}.0",
                device=device,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    parameters=parameters,
                    base_address=f"{base_address}.1",
                    device=device,
                )
            )
        return layers

    def forward(self, x):
        x = self.conv1(x)

        N, C, H, W = x.shape
        out_h = int(((H + 2 - (3 - 1) - 1) / 2) + 1)
        out_w = int(((W + 2 - (3 - 1) - 1) / 2) + 1)
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
            # is_out_tiled=False,
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
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=False,
        )
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (N, out_h, out_w, C))

        x = ttnn.permute(x, (0, 3, 1, 2))
        out = []
        for i, module in enumerate(self.layer1):
            x = module.forward(x)
        out.append(x)

        for i, module in enumerate(self.layer2):
            x = module.forward(x)

        out.append(x)

        for i, module in enumerate(self.layer3):
            x = module.forward(x)

        out.append(x)

        for i, module in enumerate(self.layer4):
            x = module.forward(x)
        out.append(x)

        return out
