# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import ttnn


@dataclass(frozen=True)
class ConvInputSpec:
    """Placement of a conv input. DRAM inputs are interleaved, L1 inputs are sharded."""

    buffer_type: ttnn.BufferType
    layout: ttnn.Layout


DRAM_ROW_MAJOR = ConvInputSpec(ttnn.BufferType.DRAM, ttnn.ROW_MAJOR_LAYOUT)
DRAM_TILE = ConvInputSpec(ttnn.BufferType.DRAM, ttnn.TILE_LAYOUT)
L1_ROW_MAJOR = ConvInputSpec(ttnn.BufferType.L1, ttnn.ROW_MAJOR_LAYOUT)
L1_TILE = ConvInputSpec(ttnn.BufferType.L1, ttnn.TILE_LAYOUT)


def conv_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class TtRTDetrConvEncoder:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        batch_size: int,
        input_height: int,
        input_width: int,
    ):
        self.model = TtRTDetrResNetBackBone(
            config=config,
            parameters=parameters.model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
        )

        self.out_indices = config.backbone_config.out_indices

    def __call__(self, pixel_values: ttnn.Tensor) -> list[tuple[ttnn.Tensor, int, int]]:
        batch_size, _, input_height, input_width = pixel_values.shape

        feature_maps = self.model(
            pixel_values,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
        )

        return feature_maps


class TtRTDetrResNetBackBone:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        batch_size: int,
        input_height: int,
        input_width: int,
    ):
        self.embeddings = TtRTDetrResNetEmbeddings(
            config=config,
            parameters=parameters.embedder,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
        )

        self.encoder = TtRTDetrResNetEncoder(
            config=config,
            parameters=parameters.encoder,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            input_height=self.embeddings.output_height,
            input_width=self.embeddings.output_width,
            input_spec=self.embeddings.output_spec,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
    ) -> list[tuple[ttnn.Tensor, int, int]]:
        x, height, width = self.embeddings(
            x,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
        )

        feature_maps = self.encoder(
            x,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
        )
        return feature_maps


class TtRTDetrResNetEmbeddings:
    # max_pool2d writes its output height-sharded in L1.
    output_spec = L1_ROW_MAJOR

    def __init__(self, config, parameters, device, dtype, batch_size: int, input_height: int, input_width: int):
        self.device = device
        self.dtype = dtype
        self.embedding_size = config.backbone_config.embedding_size
        self.hidden_act = config.backbone_config.hidden_act

        layer_configs = [
            (config.backbone_config.num_channels, self.embedding_size // 2, (2, 2)),
            (self.embedding_size // 2, self.embedding_size // 2, (1, 1)),
            (self.embedding_size // 2, self.embedding_size, (1, 1)),
        ]

        self.layers = []
        height, width = input_height, input_width
        # The input is NCHW pixels permuted to NHWC in DRAM.
        input_spec = DRAM_ROW_MAJOR
        for index, (in_channels, out_channels, stride) in enumerate(layer_configs):
            layer = TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.embedder[index],
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                activation=self.hidden_act,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                input_spec=input_spec,
            )
            self.layers.append(layer)
            height, width, input_spec = layer.output_height, layer.output_width, layer.output_spec

        self.output_height = conv_output_size(height, kernel=3, stride=2, padding=1)
        self.output_width = conv_output_size(width, kernel=3, stride=2, padding=1)

    def __call__(
        self,
        pixel_values: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
    ) -> tuple[ttnn.Tensor, int, int]:
        height = input_height
        width = input_width

        pixel_values = ttnn.permute(pixel_values, (0, 2, 3, 1))
        for layer in self.layers:
            pixel_values, height, width = layer(
                pixel_values,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
            )

        pixel_values = ttnn.max_pool2d(
            input_tensor=pixel_values,
            batch_size=batch_size,
            input_h=height,
            input_w=width,
            channels=self.embedding_size,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=False,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
        )

        return pixel_values, self.output_height, self.output_width


class TtRTDetrResNetEncoder:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        batch_size: int,
        input_height: int,
        input_width: int,
        input_spec: ConvInputSpec,
    ):
        depths = config.backbone_config.depths
        hidden_sizes = config.backbone_config.hidden_sizes
        embedding_size = config.backbone_config.embedding_size

        first_stride = (2, 2) if config.backbone_config.downsample_in_first_stage else (1, 1)
        stage_configs = [(embedding_size, hidden_sizes[0], depths[0], first_stride)]
        for index, depth in enumerate(depths[1:], start=1):
            stage_configs.append((hidden_sizes[index - 1], hidden_sizes[index], depth, (2, 2)))

        self.stages = []
        height, width = input_height, input_width
        for index, (in_channels, out_channels, depth, stride) in enumerate(stage_configs):
            stage = TtRTDetrResNetStage(
                config=config,
                parameters=parameters.stages[index],
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=out_channels,
                depth=depth,
                stride=stride,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                input_spec=input_spec,
            )
            self.stages.append(stage)
            height, width, input_spec = stage.output_height, stage.output_width, stage.output_spec

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, input_height: int, input_width: int
    ) -> list[tuple[ttnn.Tensor, int, int]]:
        feature_maps = []

        x, input_height, input_width = self.stages[0](x, batch_size, input_height, input_width)

        for stage in self.stages[1:]:
            x, input_height, input_width = stage(x, batch_size, input_height, input_width)
            x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            feature_maps.append((x_dram, input_height, input_width))

        return feature_maps


class TtRTDetrResNetStage:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: tuple[int, int],
        batch_size: int,
        input_height: int,
        input_width: int,
        input_spec: ConvInputSpec,
    ):
        self.layers = []

        height, width = input_height, input_width
        for index in range(depth):
            layer = TtRTDetrResNetBottleNeckLayer(
                config=config,
                parameters=parameters.layers[index],
                device=device,
                dtype=dtype,
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                stride=stride if index == 0 else (1, 1),
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                input_spec=input_spec,
            )
            self.layers.append(layer)
            height, width, input_spec = layer.output_height, layer.output_width, layer.output_spec

        self.output_height = height
        self.output_width = width
        self.output_spec = input_spec

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
    ) -> tuple[ttnn.Tensor, int, int]:
        height = input_height
        width = input_width

        for layer in self.layers:
            x, height, width = layer(
                x,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
            )

        return x, height, width


class TtRTDetrResNetBottleNeckLayer:
    # The residual add of two sharded L1 operands stays sharded in L1.
    output_spec = L1_TILE

    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        batch_size: int,
        input_height: int,
        input_width: int,
        input_spec: ConvInputSpec,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        reduce_channels = out_channels // 4

        layer_configs = [
            (in_channels, reduce_channels, (1, 1), (1, 1), (0, 0), "relu"),
            (reduce_channels, reduce_channels, (3, 3), stride, (1, 1), "relu"),
            (reduce_channels, out_channels, (1, 1), (1, 1), (0, 0), "identity"),
        ]

        self.layers = []
        height, width, layer_input_spec = input_height, input_width, input_spec
        for index, (layer_in, layer_out, kernel_size, layer_stride, padding, activation) in enumerate(layer_configs):
            layer = TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.layer[index],
                device=device,
                dtype=dtype,
                in_channels=layer_in,
                out_channels=layer_out,
                kernel_size=kernel_size,
                stride=layer_stride,
                padding=padding,
                activation=activation,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                input_spec=layer_input_spec,
            )
            self.layers.append(layer)
            height, width, layer_input_spec = layer.output_height, layer.output_width, layer.output_spec

        self.output_height = height
        self.output_width = width

        if "shortcut" in parameters:
            self.shortcut = TtRTDetrResNetShortcut(
                config=config,
                parameters=parameters.shortcut,
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=stride != (1, 1),
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                input_spec=input_spec,
            )
        else:
            self.shortcut = None

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, input_height: int, input_width: int
    ) -> tuple[ttnn.Tensor, int, int]:
        residual = x

        main = x
        main_height = input_height
        main_width = input_width

        for layer in self.layers:
            main, main_height, main_width = layer(
                main,
                batch_size=batch_size,
                input_height=main_height,
                input_width=main_width,
            )

        if self.shortcut is not None:
            shortcut, _, _ = self.shortcut(
                residual,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
            )
        else:
            shortcut = residual

        x = ttnn.add(main, shortcut, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])

        return x, main_height, main_width


class TtRTDetrResNetShortcut:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        batch_size: int,
        input_height: int,
        input_width: int,
        input_spec: ConvInputSpec,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if downsample:
            input_height = conv_output_size(input_height, kernel=2, stride=2, padding=0)
            input_width = conv_output_size(input_width, kernel=2, stride=2, padding=0)
            # avg_pool2d writes a sharded tiled output in L1.
            input_spec = L1_TILE

        self.projection = TtRTDetrResNetConvLayer(
            config=config,
            parameters=parameters,
            device=device,
            dtype=dtype,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation="identity",
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            input_spec=input_spec,
        )

    def __call__(
        self,
        x,
        batch_size,
        input_height,
        input_width,
    ) -> tuple[ttnn.Tensor, int, int]:
        if self.downsample:
            x = ttnn.avg_pool2d(
                input_tensor=x,
                batch_size=batch_size,
                input_h=input_height,
                input_w=input_width,
                channels=self.in_channels,
                kernel_size=[2, 2],
                stride=[2, 2],
                padding=[0, 0],
                ceil_mode=False,
                count_include_pad=True,
                divisor_override=None,
                deallocate_input=False,
                dtype=self.dtype,
                output_layout=ttnn.TILE_LAYOUT,
            )
            input_height = conv_output_size(input_height, kernel=2, stride=2, padding=0)
            input_width = conv_output_size(input_width, kernel=2, stride=2, padding=0)

        x, output_height, output_width = self.projection(
            x, batch_size=batch_size, input_height=input_height, input_width=input_width
        )

        return x, output_height, output_width


class TtRTDetrResNetConvLayer:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        activation: str,
        batch_size: int,
        input_height: int,
        input_width: int,
        input_spec: ConvInputSpec,
    ):
        self.device = device
        self.dtype = dtype
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_spec = input_spec
        self.output_height = conv_output_size(input_height, kernel_size[0], stride[0], padding[0])
        self.output_width = conv_output_size(input_width, kernel_size[1], stride[1], padding[1])

        if activation == "relu":
            self.activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        elif activation == "silu":
            self.activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        elif activation == "identity":
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation {activation}")

        is_matmul_conv = kernel_size == (1, 1) and stride == (1, 1) and padding == (0, 0)
        is_sliced_conv = input_spec.buffer_type == ttnn.BufferType.DRAM and not is_matmul_conv

        shard_layout, output_layout = None, None
        if out_channels <= 256:
            shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            output_layout = ttnn.ROW_MAJOR_LAYOUT
        else:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
            output_layout = ttnn.TILE_LAYOUT
        if is_sliced_conv:
            # Whether a DRAM conv is sliced depends on the grid, and so does its output layout
            # unless pinned. Pinning it keeps the next conv's prepared weights valid on any grid.
            output_layout = ttnn.TILE_LAYOUT

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=dtype,
            activation=self.activation,
            output_layout=output_layout,
            shard_layout=shard_layout,
            reshard_if_not_optimal=True,
        )
        # bf16 dest accumulation over the deep stage-4 reductions drops the
        # backbone output below the reference tolerance.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if is_sliced_conv:
            self.output_spec = DRAM_TILE
        elif is_matmul_conv or shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            self.output_spec = L1_TILE
        else:
            self.output_spec = L1_ROW_MAJOR

        self._prepare_weights(parameters, config.batch_norm_eps)

    def _prep_input_memory_config(self) -> ttnn.MemoryConfig:
        if self.input_spec.buffer_type == ttnn.BufferType.DRAM:
            return ttnn.DRAM_MEMORY_CONFIG
        # With reshard_if_not_optimal the conv derives its own sharding from the shapes, so weight
        # preparation only reads that the input is sharded in L1 and how its channels are aligned.
        shard_width = ttnn.core.roundup(self.in_channels, ttnn.TILE_SIZE)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [ttnn.TILE_SIZE, shard_width],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _prepare_weights(self, parameters, eps: float):
        scale = parameters.normalization.weight * torch.rsqrt(parameters.normalization.running_var + eps)
        weight = ttnn.from_torch(
            tensor=parameters.convolution.weight * scale[:, None, None, None],
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        bias = ttnn.from_torch(
            parameters.normalization.bias - parameters.normalization.running_mean * scale[None, None, None, :],
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        conv_kwargs = {
            "input_memory_config": self._prep_input_memory_config(),
            "input_layout": self.input_spec.layout,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "batch_size": self.batch_size,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "input_dtype": self.dtype,
            "conv_config": self.conv_config,
            "compute_config": self.compute_config,
        }
        weight = ttnn.prepare_conv_weights(weight_tensor=weight, weights_format="OIHW", has_bias=True, **conv_kwargs)
        bias = ttnn.prepare_conv_bias(bias_tensor=bias, **conv_kwargs)
        self.conv_weight = ttnn.to_device(weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.conv_bias = ttnn.to_device(bias, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _check_input(self, x: ttnn.Tensor, batch_size: int, input_height: int, input_width: int):
        # The weights were laid out for this exact input placement and shape.
        expected_sharded = self.input_spec.buffer_type == ttnn.BufferType.L1
        actual = (x.memory_config().buffer_type, x.is_sharded(), x.layout, batch_size, input_height, input_width)
        expected = (
            self.input_spec.buffer_type,
            expected_sharded,
            self.input_spec.layout,
            self.batch_size,
            self.input_height,
            self.input_width,
        )
        if actual != expected:
            raise ValueError(
                f"Conv {self.in_channels}->{self.out_channels} k={self.kernel_size} got input "
                f"(buffer, sharded, layout, batch, height, width)={actual}, weights were prepared for {expected}"
            )

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, input_height: int, input_width: int
    ) -> tuple[ttnn.Tensor, int, int]:
        self._check_input(x, batch_size, input_height, input_width)

        x, (output_height, output_width) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            return_output_dim=True,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(1, 1),
            groups=1,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )

        return x, output_height, output_width
