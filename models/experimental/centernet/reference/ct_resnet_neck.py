import torch
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


# outs, offset, mask, weight, self.bias)
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


class CTResNetNeck(nn.Module):
    def __init__(self, parameters=None, init_cfg=None) -> None:
        super().__init__()
        self.fp16_enabled = False
        self.parameters = parameters
        self.deconv_layers = self._make_deconv_layer()
        self.weight1 = self.parameters[f"neck.deconv_layers.{0}.conv.weight"]
        self.weight2 = self.parameters[f"neck.deconv_layers.{2}.conv.weight"]
        self.weight3 = self.parameters[f"neck.deconv_layers.{4}.conv.weight"]
        self.bias = None
        # self.stride = (1, 1)
        # self.padding = (1, 1)
        # self.dilation = (1, 1)

    def _make_deconv_layer(self) -> nn.Sequential:
        """use deconv layers to upsample backbone's output."""
        inplanes = [256, 256, 128, 128, 64, 64]
        conv_in = [512, 256, 256, 128, 128, 64]
        conv_out = [27, 256, 27, 128, 27, 64]
        kernel = 4
        stride = 1
        layers = []
        for i in range(6):
            if i % 2 == 0:
                kernel = 3
                stride = 1
            else:
                kernel = 4
                stride = 2
            if i % 2 == 0:
                conv_module = nn.Conv2d(conv_in[i], conv_out[i], kernel_size=kernel, stride=stride, padding=1)
                conv_module.weight = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.conv.conv_offset.weight"])
                conv_module.bias = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.conv.conv_offset.bias"])
            else:
                conv_module = nn.ConvTranspose2d(conv_in[i], conv_out[i], kernel_size=kernel, stride=stride, padding=1)
                conv_module.weight = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.conv.weight"])
            layers.append(conv_module)
            bn = nn.BatchNorm2d(inplanes[i]).eval()
            bn.weight = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.bn.weight"])
            bn.bias = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.bn.bias"])
            bn.running_mean = self.parameters[f"neck.deconv_layers.{i}.bn.running_mean"]
            bn.running_var = self.parameters[f"neck.deconv_layers.{i}.bn.running_var"]

            layers.append(bn)
            relu = nn.ReLU(inplace=True)
            layers.append(relu)

        return layers

    def forward(self, x) -> Tuple[torch.Tensor]:
        outs = x[-1]
        j = 3
        for i, module in enumerate(self.deconv_layers):
            if i == 0 or i == 6 or i == 12:
                temp = outs
                outs = module(outs)
                o1, o2, mask = torch.chunk(outs, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)

                mask = torch.sigmoid(mask)
                if i == 0:
                    weight = self.weight1
                elif i == 6:
                    weight = self.weight2
                elif i == 12:
                    weight = self.weight3

                outs = modulated_deform_conv(temp, offset, mask, weight, self.bias)
                j -= 1

            else:
                outs = module(outs)

        return (outs,)
