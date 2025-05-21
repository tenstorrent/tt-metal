# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.centernet.tt.common import TtConv, TtBatch_norm
import torch.nn as nn
from typing import Tuple
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext("_ext", ["modulated_deform_conv_forward", "modulated_deform_conv_backward"])


def _output_size(input, weight):
    channels = weight.size(0)
    output_size = (input.size(0), channels)
    for d in range(input.dim() - 2):
        in_size = input.size(d + 2)
        pad = 1
        kernel = 1 * (weight.size(d + 2) - 1) + 1
        stride_ = 1
        output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
    return output_size


def modulated_deform_conv(input, offset, mask, weight, bias):
    input = input.type_as(offset)
    weight = weight.type_as(input)
    bias = input.new_empty(0)
    bias = bias.type_as(input)  # type: ignore
    mask = mask.type_as(input)
    output = input.new_empty([int(i) for i in _output_size(input, weight)])
    _bufs = [input.new_empty(0), input.new_empty(0)]

    ext_module.modulated_deform_conv_forward(
        input,
        weight,
        bias,
        _bufs[0],
        offset,
        mask,
        output,
        _bufs[1],
        kernel_h=weight.size(2),
        kernel_w=weight.size(3),
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        dilation_h=1,
        dilation_w=1,
        group=1,
        deformable_group=1,
        with_bias=None,
    )

    return output


class TtCTResNetNeck:
    def __init__(self, parameters=None, init_cfg=None, device=None) -> None:
        self.parameters = parameters
        self.device = device
        self.deconv_layers = self._make_deconv_layer()
        self.weight1 = self.parameters[f"neck.deconv_layers.{0}.conv.weight"]
        self.weight2 = self.parameters[f"neck.deconv_layers.{2}.conv.weight"]
        self.weight3 = self.parameters[f"neck.deconv_layers.{4}.conv.weight"]

        self.path1 = "neck.deconv_layers.0.conv"
        self.path2 = "neck.deconv_layers.2.conv"
        self.path3 = "neck.deconv_layers.4.conv"
        self.bias = None

    def _make_deconv_layer(self) -> nn.Sequential:
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(6):
            if i % 2 == 0:
                conv_module = TtConv(
                    device=self.device,
                    parameters=self.parameters,
                    path=f"neck.deconv_layers.{i}.conv.conv_offset",
                    conv_params=[1, 1, 1, 1],
                    fused_op=True,
                    conv_transpose=False,
                    activation="",
                )
            else:
                conv_module = TtConv(
                    device=self.device,
                    parameters=self.parameters,
                    path=f"neck.deconv_layers.{i}.conv",
                    conv_params=[2, 2, 1, 1],
                    fused_op=True,
                    conv_transpose=True,
                    activation="",
                )
            layers.append(conv_module)
            bn = TtBatch_norm(self.device, f"neck.deconv_layers.{i}.bn", self.parameters)

            layers.append(bn)
            layers.append(ttnn.relu)

        return layers

    def forward(self, x) -> Tuple[torch.Tensor]:
        outs = x[-1]
        j = 3
        for i, module in enumerate(self.deconv_layers):
            if i == 0 or i == 6 or i == 12:
                temp = outs
                outs = module(outs)
                o1, o2, mask = ttnn.chunk(outs, 3, dim=1)
                offset = ttnn.concat([o1, o2], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
                mask = ttnn.sigmoid_accurate(mask, memory_config=ttnn.L1_MEMORY_CONFIG)
                if i == 0:
                    weight = self.weight1
                    temp = ttnn.to_torch(temp).float()
                    offset = ttnn.to_torch(offset).float()
                    mask = ttnn.to_torch(mask).float()
                    weight = ttnn.to_torch(weight).float()

                elif i == 6:
                    weight = self.weight2
                    temp = ttnn.to_torch(temp).float()
                    offset = ttnn.to_torch(offset).float()
                    mask = ttnn.to_torch(mask).float()
                    weight = ttnn.to_torch(weight).float()
                elif i == 12:
                    weight = self.weight3
                    temp = ttnn.to_torch(temp).float()
                    offset = ttnn.to_torch(offset).float()
                    mask = ttnn.to_torch(mask).float()
                    weight = ttnn.to_torch(weight).float()

                outs = modulated_deform_conv(temp, offset, mask, weight, self.bias)
                outs = ttnn.from_torch(outs, device=self.device, dtype=ttnn.bfloat16)
                j -= 1

            else:
                outs = module(outs)

        return (outs,)
