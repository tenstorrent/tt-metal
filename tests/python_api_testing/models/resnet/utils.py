from typing import Tuple
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset

from libs import tt_lib as ttl
from utility_functions import tt2torch_tensor, torch2tt_tensor

from libs.tt_lib.utils import (
    _nearest_32 as nearest_32,
    pad_activation,
    pad_weight,
    tilize,
    tilize_to_list,
    untilize,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
    roundup,
    roundup32,
    float_to_bits,
    divup,
    channels_last,
    convert_weights_2d_matrix
)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, state_dict=None, base_address=None) -> nn.Conv2d:
    """3x3 convolution with padding"""
    conv =  nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    conv.weight = nn.Parameter(state_dict[f"{base_address}.weight"])


    return conv


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, state_dict=None, base_address=None) -> nn.Conv2d:
    """1x1 convolution"""
    conv =  nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    conv.weight = nn.Parameter(state_dict[f"{base_address}.weight"])
    return conv


def pad_by_zero(x: torch.Tensor, device):
    initial_shape = x.shape
    if x.shape[3] % 32 != 0 or x.shape[2] % 32 != 0:
        tt_tensor = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        x = tt_tensor.pad((x.shape[0], x.shape[1], nearest_32(x.shape[2]), nearest_32(x.shape[3])), (0, 0, 0, 0), 0)
        x = x.to(ttl.tensor.Layout.TILE).to(device)

    else:
        x = torch2tt_tensor(x, device)
    return x, initial_shape

def unpad_from_zero(x, desired_shape, host):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2] :
        x = tt2torch_tensor(x)
    else:
        x = x.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
        x = x.unpad((0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1) )
        x = torch.Tensor(x.data()).reshape(x.shape())
    return x

def fold_bn_to_conv(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> Tuple[nn.Parameter]:
    # Note: this function is not used, however I am keeping it for reference
    epsilon = bn.eps # Crucially important to use batchnorm's eps

    bn_weight = bn.weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = bn.running_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = conv.weight
    weight = (conv.weight / torch.sqrt(bn_running_var + epsilon)) * bn_weight

    bn_running_mean = bn.running_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = bn.bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + epsilon)) + bn_bias

    bias = bias.squeeze(-1).squeeze(-1).squeeze(-1)

    return (nn.Parameter(weight), nn.Parameter(bias))
