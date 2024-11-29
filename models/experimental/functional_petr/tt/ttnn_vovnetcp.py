# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_petr.tt.common import Conv, Conv_with_split
import torch
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import torch.nn.functional as F


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
        self.avg_pool = ttnn.global_avg_pool2d
        if is_split:
            self.fc = Conv_with_split([1, 1, 0, 0], parameters["fc"])
        else:
            self.fc = Conv([1, 1, 0, 0], parameters["fc"])
        self.hsigmoid = ttnn_hsigmoid()

    def __call__(self, device, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(device, x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self.hsigmoid(x)
        # x = ttnn.div(ttnn.relu6(x + 3.0), 6.0)  # Hsigmoid()
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
                [1, 1, 0, 0], parameters["{}_reduction_0".format(module_name)], activation="relu"
            )
        for i in range(layer_per_block):
            module_name_with_i = "{}_{}".format(module_name, i)
            if module_name_with_i == "OSA3_1_0":
                self.layers.append(
                    Conv_with_split(
                        [1, 1, 1, 1], parameters["{}_{}".format(module_name, i)], activation="relu", split_factor=4
                    )
                )
            elif i == 0 and "OSA3_1" not in module_name_with_i and "OSA2" not in module_name_with_i:
                self.layers.append(
                    Conv_with_split(
                        [1, 1, 1, 1],
                        parameters["{}_{}".format(module_name, i)],
                        activation="relu",
                        split_factor=8,
                    )
                )
            elif (
                "OSA3_1" in module_name_with_i
                or ("OSA3_2" in module_name_with_i and module_name_with_i != "OSA3_2_0")
                or "OSA4" in module_name_with_i
                or "OSA3_3" in module_name_with_i
            ):
                self.layers.append(
                    Conv_with_split([1, 1, 1, 1], parameters["{}_{}".format(module_name, i)], activation="relu")
                )
            else:
                self.layers.append(Conv([1, 1, 1, 1], parameters["{}_{}".format(module_name, i)], activation="relu"))

        if module_name != "OSA2_1":
            if "OSA4" in module_name and module_name != "OSA4_1":
                self.conv_concat = Conv_with_split(
                    [1, 1, 0, 0], parameters["{}_{}".format(module_name, "concat")], activation="relu", split_factor=4
                )
            elif (
                module_name == "OSA3_3" or module_name == "OSA4_1" or module_name == "OSA3_1" or module_name == "OSA3_2"
            ):
                self.conv_concat = Conv_with_split(
                    [1, 1, 0, 0], parameters["{}_{}".format(module_name, "concat")], activation="relu", split_factor=16
                )
            elif "OSA5" in module_name:
                self.conv_concat = Conv_with_split(
                    [1, 1, 0, 0], parameters["{}_{}".format(module_name, "concat")], activation="relu", split_factor=16
                )
            else:
                self.conv_concat = Conv(
                    [1, 1, 0, 0], parameters["{}_{}".format(module_name, "concat")], activation="relu"
                )
        if module_name == "OSA5_1" or module_name == "OSA5_2" or module_name == "OSA5_3":
            self.ese = ttnn_esemodule(parameters, is_split=True)
        else:
            self.ese = ttnn_esemodule(parameters)

    def __call__(self, device, x):
        identity_feat = x
        output = []
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        output.append(x)

        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for i, layer in enumerate(self.layers):
            module_name_with_i = "{}_{}".format(self.module_name, i)

            if "OSA2_1" in module_name_with_i:
                x = ttnn.permute(x, (0, 3, 1, 2))
                x = ttnn.to_torch(x)
                if x.dtype == torch.bfloat16:
                    x = x.to(torch.float)
                x = F.conv2d(
                    x,
                    self.parameters["{}_{}".format(self.module_name, i)]["weight"],
                    bias=self.parameters["{}_{}".format(self.module_name, i)]["bias"],
                    stride=1,
                    padding=1,
                )
                x = ttnn.from_torch(x.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
                x = ttnn.relu(x)
                if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            else:
                x = layer(device, x)
                if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            output.append(x)
        x = ttnn.concat(output, dim=3)

        for y in output:
            ttnn.deallocate(y)
        if self.module_name != "OSA2_1":
            x = self.conv_concat(device, x)
        else:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.to_torch(x)
            x = torch.permute(x, (0, 3, 1, 2))
            x = torch_to_tt_tensor_rm(x, device, put_on_device=True)
            xt = fallback_ops.conv2d(
                x,
                self.parameters["{}_{}".format(self.module_name, "concat")]["weight"],
                self.parameters["{}_{}".format(self.module_name, "concat")]["bias"],
                1,
                0,
                1,
                1,
            )
            xt = tt_to_torch_tensor(xt)
            xt = ttnn.from_torch(xt.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            x = ttnn.relu(xt)

        x = self.ese(device, x)
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if self.identity:
            x = x + identity_feat

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
            x = ttnn.permute(x, (0, 3, 1, 2))
            x = ttnn.to_torch(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
            x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16)
            if x.get_layout() != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.permute(x, (0, 2, 3, 1))

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

        self.stage2 = ttnn_osa_stage(parameters.stage2, 128, 128, 256, 1, 5, 2, SE=True, depthwise=False)
        self.stage3 = ttnn_osa_stage(parameters.stage3, 256, 160, 512, 3, 5, 3, SE=True, depthwise=False)
        self.stage4 = ttnn_osa_stage(parameters.stage4, 512, 192, 768, 9, 5, 4, SE=True, depthwise=False)
        self.stage5 = ttnn_osa_stage(parameters.stage5, 768, 224, 1024, 3, 5, 5, SE=True, depthwise=False)

    def __call__(self, device, x):
        outputs = []
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.to_torch(x)
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float)
        x = F.conv2d(
            x,
            self.stem_parameters["stem_1"]["weight"],
            bias=self.stem_parameters["stem_1"]["bias"],
            stride=2,
            padding=1,
        )
        x = F.conv2d(
            x,
            self.stem_parameters["stem_2"]["weight"],
            bias=self.stem_parameters["stem_2"]["bias"],
            stride=1,
            padding=1,
        )
        stem = F.conv2d(
            x,
            self.stem_parameters["stem_3"]["weight"],
            bias=self.stem_parameters["stem_3"]["bias"],
            stride=2,
            padding=1,
        )
        x = ttnn.from_torch(stem.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        stage2 = self.stage2(device, x)
        stage3 = self.stage3(device, stage2)
        stage4 = self.stage4(device, stage3)
        stage5 = self.stage5(device, stage4)

        tensors = {"stem": stem, "stage2": stage2, "stage3": stage3, "stage4": stage4, "stage5": stage5}

        for name, tensor in tensors.items():
            if name in self.out_features:
                outputs.append(tensor)
        return outputs
