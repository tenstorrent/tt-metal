# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.petr.tt.common import Conv, Conv_with_split
from models.tt_cnn.tt.builder import TtMaxPool2d, MaxPool2dConfiguration
from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration
import torch
from loguru import logger


class ttnn_hsigmoid:
    def __init__(self, inplace=True):
        self.inplace = inplace

    def __call__(self, x):
        x = x + 3.0
        x = ttnn.relu6(x)
        x = ttnn.div(x, 6.0)
        return x


class ttnn_eSEModule:
    def __init__(self, parameters, model_config=None, conv_args=None, device=None, is_split=False):
        if is_split or conv_args is None:
            self.fc = Conv_with_split([1, 1, 0, 0], parameters)
        else:
            self.fc_config = Conv2dConfiguration.from_model_args(
                conv2d_args=conv_args,
                weights=parameters["weight"],
                bias=parameters["bias"],
                math_fidelity=model_config["MATH_FIDELITY"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                activation_dtype=model_config["ACTIVATIONS_DTYPE"],
            )
            self.fc = TtConv2d(self.fc_config, device)
        self.hsigmoid = ttnn_hsigmoid()

    def __call__(self, device, x):
        input = x
        B, H, W, C = x.shape
        x = ttnn.reshape(x, (B, H * W, C))
        x = ttnn.mean(x, dim=1, keepdim=True)
        x = ttnn.reshape(x, (B, 1, 1, C))
        if hasattr(self, "fc_config"):
            x, [out_h, out_w] = self.fc(x, return_output_dim=True)
            x = ttnn.reshape(x, (self.fc_config.batch_size, out_h, out_w, self.fc_config.out_channels))
        else:
            x, [out_h, out_w] = self.fc(device, x)
            out_channels = x.shape[-1]
            x = ttnn.reshape(x, (B, out_h, out_w, out_channels))
        x = self.hsigmoid(x)
        if input.get_layout() != ttnn.TILE_LAYOUT:
            input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
        return input * x


class ttnn_OSA_module:
    def __init__(
        self,
        parameters,
        in_ch,
        stage_ch,
        concat_ch,
        layer_per_block,
        module_name,
        SE=False,
        identity=False,
        depthwise=False,
        with_cp=True,
        model_config=None,
        conv_args=None,
        device=None,
    ):
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.parameters = parameters
        self.module_name = module_name
        self.layers = []
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = Conv(
                [1, 1, 0, 0], parameters["{}_reduction_0".format(module_name)], activation="relu", height_sharding=True
            )
        for i in range(layer_per_block):
            module_name_with_i = "{}_{}".format(module_name, i)
            if module_name_with_i == "OSA3_1_0":
                self.layers.append(
                    Conv(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        act_block_h=128,
                        height_sharding=False,
                    )
                )
            elif i == 0 and "OSA3_1" not in module_name_with_i and "OSA2" not in module_name_with_i:
                self.layers.append(
                    Conv(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        act_block_h=128,
                        height_sharding=False,
                    )
                )
            elif (
                "OSA3_1" in module_name_with_i
                or ("OSA3_2" in module_name_with_i and module_name_with_i != "OSA3_2_0")
                or "OSA4" in module_name_with_i
                or "OSA3_3" in module_name_with_i
            ):
                self.layers.append(
                    Conv(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        act_block_h=128,
                        height_sharding=False,
                    )
                )
            else:
                self.layers.append(
                    Conv(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        height_sharding=False,
                    )
                )

        if module_name != "OSA2_1":
            if "OSA4" in module_name and module_name != "OSA4_1":
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=False,
                )
            elif module_name == "OSA4_1":
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=False,
                )
            elif "OSA5" in module_name:
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=False,
                )
            else:
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=False,
                )
        ese_conv_args = None
        if conv_args is not None:
            try:
                if isinstance(conv_args, dict):
                    if module_name in conv_args:
                        module_args = conv_args[module_name]
                        if isinstance(module_args, dict) and "ese" in module_args:
                            ese_conv_args = module_args["ese"].get("fc")
                        elif hasattr(module_args, "ese"):
                            ese_conv_args = getattr(module_args.ese, "fc", None)
                elif hasattr(conv_args, module_name):
                    module_args = getattr(conv_args, module_name)
                    if hasattr(module_args, "ese"):
                        ese_conv_args = getattr(module_args.ese, "fc", None)
            except (KeyError, AttributeError):
                pass
            if ese_conv_args is not None and not hasattr(ese_conv_args, "input_height"):
                ese_conv_args = None
        if module_name == "OSA5_1" or module_name == "OSA5_2" or module_name == "OSA5_3":
            self.ese = ttnn_eSEModule(parameters["fc"], model_config, ese_conv_args, device, is_split=True)
        else:
            self.ese = ttnn_eSEModule(parameters["fc"], model_config, ese_conv_args, device)

    def __call__(self, device, x):
        identity_feat = x
        output = []
        input_shape = x.shape
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            if hasattr(x, "memory_config") and x.memory_config().is_sharded():
                if x.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
                    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        output.append(x)

        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)

        for i, layer in enumerate(self.layers):
            module_name_with_i = "{}_{}".format(self.module_name, i)

            if "OSA2_1" in module_name_with_i:
                conv_layer = Conv([1, 1, 1, 1], self.parameters["{}_{}".format(self.module_name, i)], activation="relu")
                x = conv_layer(device, x)
            else:
                x = layer(device, x)
            if hasattr(x, "memory_config") and x.memory_config().is_sharded():
                if x.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
                    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            output.append(x)
        output_tensor = []
        for idx in range(len(output)):
            if hasattr(output[idx], "memory_config") and output[idx].memory_config().is_sharded():
                if output[idx].memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
                    output[idx] = ttnn.to_memory_config(output[idx], ttnn.DRAM_MEMORY_CONFIG)
            if output[idx].get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                output[idx] = ttnn.to_layout(output[idx], ttnn.ROW_MAJOR_LAYOUT)
            output_torch = ttnn.to_torch(output[idx]).to(torch.bfloat16)
            output_tensor.append(ttnn.from_torch(output_torch, dtype=ttnn.bfloat16, device=device))

        x = ttnn.concat(output_tensor, dim=3)

        if self.module_name != "OSA2_1":
            x = self.conv_concat(device, x)
        else:
            conv_layer = Conv(
                [1, 1, 0, 0],
                self.parameters["{}_{}".format(self.module_name, "concat")],
                activation="relu",
            )
            x = conv_layer(device, x)

        x = self.ese(device, x)
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if self.identity:
            x = x + identity_feat

        if len(x.shape) != 4:
            logger.error(f"ERROR: {self.module_name} produced non-4D tensor: {x.shape}")

        if len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] > 100:
            batch_size = x.shape[0]
            channels = x.shape[3]
            total_spatial = x.shape[2]

            if "OSA2" in self.module_name:
                height, width = 80, 200
            elif "OSA3" in self.module_name:
                height, width = 40, 100
            elif "OSA4" in self.module_name:
                height, width = 20, 50
            elif "OSA5" in self.module_name:
                height, width = 10, 25
            else:
                if self.identity and len(input_shape) == 4:
                    height, width = input_shape[1], input_shape[2]
                else:
                    height = int(total_spatial**0.5)
                    width = total_spatial // height

            if height * width == total_spatial:
                x = ttnn.reshape(x, (batch_size, height, width, channels))

        return x


class ttnn_OSA_stage:
    def __init__(
        self,
        parameters,
        in_ch,
        stage_ch,
        concat_ch,
        block_per_stage,
        layer_per_block,
        stage_num,
        SE=False,
        depthwise=False,
        model_config=None,
        conv_args=None,
        device=None,
        torch_fallback=True,
    ):
        self.torch_fallback = torch_fallback
        self.blocks = []
        if not stage_num == 2:
            self.pooling = True
        else:
            self.pooling = False

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        setattr(
            self,
            module_name,
            ttnn_OSA_module(
                parameters[module_name],
                in_ch,
                stage_ch,
                concat_ch,
                layer_per_block,
                module_name,
                SE,
                depthwise=depthwise,
                model_config=model_config,
                conv_args=conv_args,
                device=device,
            ),
        )
        self.blocks.append(module_name)

        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            setattr(
                self,
                module_name,
                ttnn_OSA_module(
                    parameters[module_name],
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise,
                    model_config=model_config,
                    conv_args=conv_args,
                    device=device,
                ),
            )
            self.blocks.append(module_name)

    def __call__(self, device, x):
        if self.pooling is True:
            if self.torch_fallback:
                x_torch = ttnn.to_torch(x).to(torch.bfloat16)
                x_torch = x_torch.permute(0, 3, 1, 2)
                x_torch = torch.nn.functional.max_pool2d(x_torch, kernel_size=3, stride=2, padding=0, ceil_mode=True)
                x_torch = x_torch.permute(0, 2, 3, 1)
                x = ttnn.from_torch(x_torch.to(torch.float32), dtype=ttnn.bfloat16, device=device)
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            else:
                batch_size, input_height, input_width, channels = x.shape
                pool_config = MaxPool2dConfiguration(
                    input_height=input_height,
                    input_width=input_width,
                    channels=channels,
                    batch_size=batch_size,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(0, 0),
                    dilation=(1, 1),
                    ceil_mode=True,
                    dtype=ttnn.bfloat16,
                    output_layout=ttnn.TILE_LAYOUT,
                )
                maxpool = TtMaxPool2d(pool_config, device)
                x = maxpool(x)

        for module_name in self.blocks:
            module = getattr(self, module_name)
            x = module(device, x)

        return x


class ttnn_VoVNetCP:
    def __init__(
        self,
        parameters,
        stem_parameters,
        device,
        out_features=["stage5", "stage4"],
    ):
        self.device = device
        self.stem_parameters = stem_parameters
        self.out_features = out_features

        # Initialize stem convolutions using Conv class
        self.stem_conv1 = Conv([2, 2, 1, 1], stem_parameters["stem_1"], activation="relu", height_sharding=False)

        self.stem_conv2 = Conv([1, 1, 1, 1], stem_parameters["stem_2"], activation="relu", height_sharding=False)

        self.stem_conv3 = Conv([2, 2, 1, 1], stem_parameters["stem_3"], activation="relu", height_sharding=False)

        default_model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }

        self.stage2 = ttnn_OSA_stage(
            parameters.stage2,
            128,
            128,
            256,
            1,
            5,
            2,
            SE=True,
            depthwise=False,
            model_config=default_model_config,
            conv_args=None,
            device=device,
        )
        self.stage3 = ttnn_OSA_stage(
            parameters.stage3,
            256,
            160,
            512,
            3,
            5,
            3,
            SE=True,
            depthwise=False,
            model_config=default_model_config,
            conv_args=None,
            device=device,
        )
        self.stage4 = ttnn_OSA_stage(
            parameters.stage4,
            512,
            192,
            768,
            9,
            5,
            4,
            SE=True,
            depthwise=False,
            model_config=default_model_config,
            conv_args=None,
            device=device,
        )
        self.stage5 = ttnn_OSA_stage(
            parameters.stage5,
            768,
            224,
            1024,
            3,
            5,
            5,
            SE=True,
            depthwise=False,
            model_config=default_model_config,
            conv_args=None,
            device=device,
        )

    def __call__(self, device, x):
        outputs = []

        # Convert to NCHW for processing
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Stem conv 1
        x = self.stem_conv1(device, x)
        # Stem conv 2
        x = self.stem_conv2(device, x)

        # Stem conv 3
        x = self.stem_conv3(device, x)

        stage2 = self.stage2(device, x)
        stage3 = self.stage3(device, stage2)
        stage4 = self.stage4(device, stage3)
        stage5 = self.stage5(device, stage4)

        tensors = {"stem": x, "stage2": stage2, "stage3": stage3, "stage4": stage4, "stage5": stage5}

        for name, tensor in tensors.items():
            if name in self.out_features:
                outputs.append(tensor)
        return outputs
