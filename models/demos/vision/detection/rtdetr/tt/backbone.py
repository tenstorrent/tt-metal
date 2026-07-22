import torch

import ttnn


class TtRTDetrConvEncoder:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
    ):
        self.model = TtRTDetrResNetBackBone(
            config=config,
            parameters=parameters.model,
            device=device,
            dtype=dtype,
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

        return [feature_maps[index - 1] for index in self.out_indices]


class TtRTDetrResNetBackBone:
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
    ):
        self.embeddings = TtRTDetrResNetEmbeddings(
            config=config,
            parameters=parameters.embedder,
            device=device,
            dtype=dtype,
        )

        self.encoder = TtRTDetrResNetEncoder(
            config=config,
            parameters=parameters.encoder,
            device=device,
            dtype=dtype,
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
    def __init__(self, config, parameters, device, dtype):
        self.device = device
        self.dtype = dtype
        self.embedding_size = config.backbone_config.embedding_size
        self.hidden_act = config.backbone_config.hidden_act

        self.layers = [
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.embedder[0],
                device=device,
                dtype=dtype,
                in_channels=config.backbone_config.num_channels,
                out_channels=self.embedding_size // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                activation=self.hidden_act,
            ),
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.embedder[1],
                device=device,
                dtype=dtype,
                in_channels=self.embedding_size // 2,
                out_channels=self.embedding_size // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation=self.hidden_act,
            ),
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.embedder[2],
                device=device,
                dtype=dtype,
                in_channels=self.embedding_size // 2,
                out_channels=self.embedding_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation=self.hidden_act,
            ),
        ]

    def __call__(
        self, pixel_values: ttnn.Tensor, batch_size: int, input_height: int, input_width: int
    ) -> tuple[ttnn.Tensor, int, int]:
        height = input_height
        width = input_width

        pixel_values = ttnn.permute(pixel_values, (0, 2, 3, 1))
        for layer in self.layers:
            pixel_values, height, width = layer(
                pixel_values, batch_size=batch_size, input_height=height, input_width=width
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

        height = (height + 2 - 3) // 2 + 1
        width = (width + 2 - 3) // 2 + 1

        return pixel_values, height, width


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
    ):
        self.device = device
        self.dtype = dtype
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if activation == "relu":
            self.activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        elif activation == "identity":
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation {activation}")

        eps = config.batch_norm_eps

        scale = parameters.normalization.weight * torch.rsqrt(parameters.normalization.running_var + eps)

        self.conv_weight = ttnn.from_torch(
            tensor=parameters.convolution.weight * scale[:, None, None, None],
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.conv_bias = ttnn.from_torch(
            parameters.normalization.bias - parameters.normalization.running_mean * scale[None, None, None, :],
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if out_channels <= 256 else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        output_layout = (
            ttnn.ROW_MAJOR_LAYOUT if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else ttnn.TILE_LAYOUT
        )

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=dtype,
            activation=self.activation,
            output_layout=output_layout,
            shard_layout=shard_layout,
            reshard_if_not_optimal=True,
        )

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, input_height: int, input_width: int
    ) -> tuple[ttnn.Tensor, int, int]:
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
        )

        return x, output_height, output_width


class TtRTDetrResNetEncoder:
    def __init__(self, config, parameters, device, dtype):
        depths = config.backbone_config.depths
        hidden_sizes = config.backbone_config.hidden_sizes
        embedding_size = config.backbone_config.embedding_size

        self.stages = []

        self.stages.append(
            TtRTDetrResNetStage(
                config=config,
                parameters=parameters.stages[0],
                device=device,
                dtype=dtype,
                in_channels=embedding_size,
                out_channels=hidden_sizes[0],
                depth=depths[0],
                stride=(2, 2) if config.backbone_config.downsample_in_first_stage else (1, 1),
            )
        )

        for index, depth in enumerate(depths[1:], start=1):
            self.stages.append(
                TtRTDetrResNetStage(
                    config=config,
                    parameters=parameters.stages[index],
                    device=device,
                    dtype=dtype,
                    in_channels=hidden_sizes[index - 1],
                    out_channels=hidden_sizes[index],
                    depth=depth,
                    stride=(2, 2),
                )
            )

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, input_height: int, input_width: int
    ) -> list[tuple[ttnn.Tensor, int, int]]:
        feature_maps = []

        for stage in self.stages:
            x, input_height, input_width = stage(x, batch_size, input_height, input_width)
            feature_maps.append((x, input_height, input_width))

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
    ):
        self.layers = []

        for index in range(depth):
            i_c = in_channels if index == 0 else out_channels
            st = stride if index == 0 else (1, 1)
            self.layers.append(
                TtRTDetrResNetBottleNeckLayer(
                    config=config,
                    parameters=parameters.layers[index],
                    device=device,
                    dtype=dtype,
                    in_channels=i_c,
                    out_channels=out_channels,
                    stride=st,
                )
            )

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
    def __init__(
        self,
        config,
        parameters,
        device,
        dtype,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        reduce_channels = out_channels // 4

        self.layers = [
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.layer[0],
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=reduce_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation="relu",
            ),
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.layer[1],
                device=device,
                dtype=dtype,
                in_channels=reduce_channels,
                out_channels=reduce_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                activation="relu",
            ),
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.layer[2],
                device=device,
                dtype=dtype,
                in_channels=reduce_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation="identity",
            ),
        ]

        if "shortcut" in parameters:
            self.shortcut = TtRTDetrResNetShortcut(
                config=config,
                parameters=parameters.shortcut,
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=stride != (1, 1),
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

        x = ttnn.add(main, shortcut)
        x = ttnn.relu(x)

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
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

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
            input_height = (input_height - 2) // 2 + 1
            input_width = (input_width - 2) // 2 + 1

        x, output_height, output_width = self.projection(
            x, batch_size=batch_size, input_height=input_height, input_width=input_width
        )

        return x, output_height, output_width
