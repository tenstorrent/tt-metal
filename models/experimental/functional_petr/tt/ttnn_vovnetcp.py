# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_petr.tt.common import Conv, Conv_with_split
import torch.nn.functional as F
import torch


class ttnn_hsigmoid:
    def __init__(self, inplace=True):
        self.inplace = inplace

    def __call__(self, x):
        x = x + 3.0
        x = ttnn.relu6(x)
        x = ttnn.div(x, 6.0)
        return x


class ttnn_esemodule:
    def __init__(self, parameters, is_split=False):
        # self.avg_pool = ttnn.global_avg_pool2d
        if is_split:
            self.fc = Conv_with_split([1, 1, 0, 0], parameters["fc"])
        else:
            self.fc = Conv([1, 1, 0, 0], parameters["fc"])
        self.hsigmoid = ttnn_hsigmoid()

    def __call__(self, device, x):
        input = x
        # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        # x = self.avg_pool(x)
        # x = ttnn.global_avg_pool2d(x)
        x_torch = ttnn.to_torch(x).to(torch.float32)  # Convert to float32
        x_torch = x_torch.mean(dim=(1, 2), keepdim=True)  # Global avg pool
        x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self.fc(device, x)
        x = self.hsigmoid(x)
        if input.get_layout() != ttnn.TILE_LAYOUT:
            input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
        return input * x


class ttnn_osa_module:
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
                        height_sharding=True,
                    )
                )
            elif i == 0 and "OSA3_1" not in module_name_with_i and "OSA2" not in module_name_with_i:
                self.layers.append(
                    Conv(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        act_block_h=128,
                        height_sharding=True,
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
                        height_sharding=True,
                    )
                )
            else:
                self.layers.append(
                    Conv(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        height_sharding=True,
                    )
                )

        if module_name != "OSA2_1":
            if "OSA4" in module_name and module_name != "OSA4_1":
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=True,
                )
            elif module_name == "OSA4_1":
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=True,
                )
            elif "OSA5" in module_name:
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=True,
                )
            else:
                self.conv_concat = Conv(
                    [1, 1, 0, 0],
                    parameters["{}_{}".format(module_name, "concat")],
                    activation="relu",
                    height_sharding=True,
                )
        if module_name == "OSA5_1" or module_name == "OSA5_2" or module_name == "OSA5_3":
            self.ese = ttnn_esemodule(parameters, is_split=True)
        else:
            self.ese = ttnn_esemodule(parameters)

    def __call__(self, device, x):
        identity_feat = x
        output = []
        input_shape = x.shape
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
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
            if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            if hasattr(x, "memory_config") and x.memory_config().is_sharded():
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            output.append(x)
        output_float32 = []
        for idx in range(len(output)):
            if output[idx].get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                output[idx] = ttnn.to_layout(output[idx], ttnn.ROW_MAJOR_LAYOUT)
            if hasattr(output[idx], "memory_config") and output[idx].memory_config().is_sharded():
                output[idx] = ttnn.to_memory_config(output[idx], ttnn.L1_MEMORY_CONFIG)
            output_torch = ttnn.to_torch(output[idx]).to(torch.float32)
            output_float32.append(ttnn.from_torch(output_torch, dtype=ttnn.bfloat16, device=device))

        x = ttnn.concat(output_float32, dim=3)

        for y in output:
            ttnn.deallocate(y)
        if self.module_name != "OSA2_1":
            x = self.conv_concat(device, x)
        else:
            conv_layer = Conv(
                [1, 1, 0, 0], self.parameters["{}_{}".format(self.module_name, "concat")], activation="relu"
            )
            x = conv_layer(device, x)

        x = self.ese(device, x)
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if self.identity:
            x = x + identity_feat

        # Ensure we're maintaining 4D shape
        if len(x.shape) != 4:
            print(f"ERROR: {self.module_name} produced non-4D tensor: {x.shape}")

        if len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] > 100:
            batch_size = x.shape[0]
            channels = x.shape[3]
            total_spatial = x.shape[2]

            # Determine expected dimensions based on module name
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


class ttnn_osa_stage:
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
    ):
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
            ttnn_osa_module(
                parameters[module_name],
                in_ch,
                stage_ch,
                concat_ch,
                layer_per_block,
                module_name,
                SE,
                depthwise=depthwise,
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
                ttnn_osa_module(
                    parameters[module_name],
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise,
                ),
            )
            self.blocks.append(module_name)

    def __call__(self, device, x):
        if self.pooling is True:
            original_dtype = x.dtype if hasattr(x, "dtype") else ttnn.bfloat16

            # Maxpool in torch
            x_torch = ttnn.to_torch(x).to(torch.float32)  # ← Use float32 in torch

            # NHWC → NCHW
            x_torch = x_torch.permute(0, 3, 1, 2)

            # Apply pooling
            x_torch = F.max_pool2d(x_torch, kernel_size=3, stride=2, padding=0, ceil_mode=True)

            # NCHW → NHWC
            x_torch = x_torch.permute(0, 2, 3, 1)

            x = ttnn.from_torch(x_torch.to(torch.float32), dtype=ttnn.bfloat16, device=device)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        for module_name in self.blocks:
            module = getattr(self, module_name)  # Retrieve the block by name
            x = module(device, x)  # Forward pass through each `ttnn_osa_module`

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
        self.stem_conv1 = Conv(
            [2, 2, 1, 1], stem_parameters["stem_1"], activation="relu", height_sharding=False  # stride=2, padding=1
        )

        self.stem_conv2 = Conv(
            [1, 1, 1, 1], stem_parameters["stem_2"], activation="relu", height_sharding=False  # stride=1, padding=1
        )

        self.stem_conv3 = Conv(
            [2, 2, 1, 1], stem_parameters["stem_3"], activation="relu", height_sharding=False  # stride=2, padding=1
        )

        # Initialize stages
        self.stage2 = ttnn_osa_stage(parameters.stage2, 128, 128, 256, 1, 5, 2, SE=True, depthwise=False)
        self.stage3 = ttnn_osa_stage(parameters.stage3, 256, 160, 512, 3, 5, 3, SE=True, depthwise=False)
        self.stage4 = ttnn_osa_stage(parameters.stage4, 512, 192, 768, 9, 5, 4, SE=True, depthwise=False)
        self.stage5 = ttnn_osa_stage(parameters.stage5, 768, 224, 1024, 3, 5, 5, SE=True, depthwise=False)

        self.stage2 = ttnn_osa_stage(parameters.stage2, 128, 128, 256, 1, 5, 2, SE=True, depthwise=False)
        self.stage3 = ttnn_osa_stage(parameters.stage3, 256, 160, 512, 3, 5, 3, SE=True, depthwise=False)
        self.stage4 = ttnn_osa_stage(parameters.stage4, 512, 192, 768, 9, 5, 4, SE=True, depthwise=False)
        self.stage5 = ttnn_osa_stage(parameters.stage5, 768, 224, 1024, 3, 5, 5, SE=True, depthwise=False)

    def __call__(self, device, x):
        outputs = []

        # Convert to NCHW for processing
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Stem conv 1 (stride=2, padding=1)
        x = self.stem_conv1(device, x)
        # Stem conv 2 (stride=1, padding=1)
        x = self.stem_conv2(device, x)

        # Stem conv 3 (stride=2, padding=1)
        x = self.stem_conv3(device, x)

        # Now x is ready for stages
        stage2 = self.stage2(device, x)
        stage3 = self.stage3(device, stage2)
        stage4 = self.stage4(device, stage3)
        stage5 = self.stage5(device, stage4)

        tensors = {"stem": x, "stage2": stage2, "stage3": stage3, "stage4": stage4, "stage5": stage5}

        for name, tensor in tensors.items():
            if name in self.out_features:
                outputs.append(tensor)
        return outputs
