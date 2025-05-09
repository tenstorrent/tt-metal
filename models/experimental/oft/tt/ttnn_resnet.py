import ttnn
import torch
import torch.nn as nn


class Conv:
    def __init__(
        self,
        model,
        path,
        input_params,
        *,
        stride=1,
        padding=1,
        has_bias=False,
        act_block_h=32,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        activation="",
        width_sharding=False,
        block_sharding=False,
    ) -> None:
        weight = model[path + ".weight"]
        self.weights = ttnn.from_torch(weight)

        self.has_bias = has_bias
        if self.has_bias:
            bias = model[path + ".bias"]
            bias = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias)

        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = stride
        self.padding = padding
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif height_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif block_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=self.shard_layout,
            # input_channels_alignment=16 if self.input_params[3] < 16 else 32,
            deallocate_activation=self.deallocate,
            activation=self.activation,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
        )
        return output_tensor, out_h, out_w


class BasicBlock:
    expansion = 1

    def __init__(self, device, model, path, input_params, inplanes, planes, stride=1, height_sharding=True):
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model

        self.input_params = input_params
        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = Conv(
            torch_model, path + ".conv1", self.input_params[0], stride=stride, height_sharding=height_sharding
        )
        self.gn1 = nn.GroupNorm(16, planes)
        self.gn1.weight = nn.Parameter(torch_model[path + ".bn1.weight"])
        self.gn1.bias = nn.Parameter(torch_model[path + ".bn1.bias"])

        self.conv2 = Conv(torch_model, path + ".conv2", self.input_params[1], height_sharding=height_sharding)
        self.gn2 = nn.GroupNorm(16, planes)
        self.gn2.weight = nn.Parameter(torch_model[path + ".bn2.weight"])
        self.gn2.bias = nn.Parameter(torch_model[path + ".bn2.bias"])

        if stride != 1 or inplanes != planes:
            self.downsample = True

            self.downsample_conv = Conv(
                torch_model,
                path + ".downsample.0",
                self.input_params[0],
                stride=stride,
                padding=0,
                height_sharding=height_sharding,
            )
            self.downsample_gn = nn.GroupNorm(16, planes)
            self.downsample_gn.weight = nn.Parameter(torch_model[path + ".downsample.1.weight"])
            self.downsample_gn.bias = nn.Parameter(torch_model[path + ".downsample.1.bias"])
        else:
            self.downsample = None

    def __call__(self, device, x):
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        identity = x
        N = x.shape[0]
        out, out_h, out_w = self.conv1(device, x)

        out = ttnn.to_torch(out)
        out = out.reshape(N, out_h, out_w, out.shape[-1])
        out = torch.permute(out, (0, 3, 1, 2))

        out = self.gn1(out)
        out = ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        out = ttnn.relu(out)
        out = ttnn.permute(out, [0, 2, 3, 1])
        out, out_h, out_w = self.conv2(device, out)
        out = ttnn.to_torch(out)

        out = out.reshape(N, out_h, out_w, out.shape[-1])
        out = torch.permute(out, (0, 3, 1, 2))

        out = self.gn2(out)

        out = ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.permute(out, [0, 2, 3, 1])

        if self.downsample is not None:
            identity, out_h, out_w = self.downsample_conv(device, x)

            identity = ttnn.to_torch(identity)

            identity = identity.reshape(N, out_h, out_w, identity.shape[-1])
            identity = torch.permute(identity, (0, 3, 1, 2))
            identity = self.downsample_gn(identity)
            identity = ttnn.from_torch(identity, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            identity = ttnn.permute(identity, [0, 2, 3, 1])

        out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)

        out += identity

        out = ttnn.relu(out)

        return out


class ResNetFeatures:
    def __init__(self, device, model, path, block, layers):
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model

        self.inplanes = 64

        self.conv1 = Conv(torch_model, path + ".conv1", [1, 370, 1224, 3], stride=2, padding=3)
        self.gn1 = nn.GroupNorm(16, 64)
        self.gn1.weight = nn.Parameter(torch_model[path + ".bn1.weight"])
        self.gn1.bias = nn.Parameter(torch_model[path + ".bn1.bias"])

        self.layer1 = self._make_layer(
            device,
            model,
            path + ".layer1",
            [([1, 93, 306, 64], [1, 93, 306, 64]), ([1, 93, 306, 64], [1, 93, 306, 64])],
            block,
            64,
            layers[0],
        )
        self.layer2 = self._make_layer(
            device,
            model,
            path + ".layer2",
            [([1, 93, 306, 64], [1, 47, 153, 128]), ([1, 47, 153, 128], [1, 47, 153, 128])],
            block,
            128,
            layers[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            device,
            model,
            path + ".layer3",
            [([1, 47, 153, 128], [1, 24, 77, 256]), ([1, 24, 77, 256], [1, 24, 77, 256])],
            block,
            256,
            layers[2],
            stride=2,
            height_sharding=False,
        )
        self.layer4 = self._make_layer(
            device,
            model,
            path + ".layer4",
            [([1, 24, 77, 256], [1, 12, 39, 512]), ([1, 12, 39, 512], [1, 12, 39, 512])],
            block,
            512,
            layers[3],
            stride=2,
            height_sharding=False,
        )

    def _make_layer(self, device, model, path, input_params, block, planes, blocks, stride=1, height_sharding=True):
        layers = []
        layers.append(
            block(
                device,
                model,
                path + ".0",
                input_params[0],
                self.inplanes,
                planes,
                stride,
                height_sharding=height_sharding,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    device,
                    model,
                    path + f".{i}",
                    input_params[i],
                    self.inplanes,
                    planes,
                    height_sharding=height_sharding,
                )
            )

        return layers

    def _run_layer(self, device, x, layer):
        for block in layer:
            x = block(device, x)
        return x

    def __call__(self, device, x):
        conv1, out_h, out_w = self.conv1(device, x)

        conv1 = ttnn.to_torch(conv1)
        conv1 = conv1.reshape(conv1.shape[0], out_h, out_w, conv1.shape[-1])
        conv1 = torch.permute(conv1, (0, 3, 1, 2))

        conv1 = self.gn1(conv1)

        conv1 = ttnn.from_torch(conv1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        conv1 = ttnn.relu(conv1)

        shape = (1, 1, conv1.shape[0] * conv1.shape[2] * conv1.shape[3], conv1.shape[1])
        conv1 = ttnn.permute(conv1, [0, 2, 3, 1])

        conv1_in = ttnn.reshape(conv1, shape)
        conv1_in = ttnn.to_layout(conv1_in, ttnn.ROW_MAJOR_LAYOUT)
        conv1_in = ttnn.max_pool2d(
            input_tensor=conv1_in,
            batch_size=conv1.shape[0],
            input_h=conv1.shape[1],
            input_w=conv1.shape[2],
            channels=conv1.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )
        conv1_out_h, conv1_out_w = int((conv1.shape[1] + 1) / 2), int((conv1.shape[2] + 1) / 2)
        conv1 = ttnn.reshape(conv1_in, (conv1.shape[0], conv1_out_h, conv1_out_w, conv1.shape[3]))

        conv1 = ttnn.sharded_to_interleaved(conv1, ttnn.L1_MEMORY_CONFIG)

        feats4 = self._run_layer(device, conv1, self.layer1)
        ttnn.deallocate(conv1)
        feats8 = self._run_layer(device, feats4, self.layer2)
        ttnn.deallocate(feats4)
        feats16 = self._run_layer(device, feats8, self.layer3)
        feats32 = self._run_layer(device, feats16, self.layer4)

        return feats8, feats16, feats32
