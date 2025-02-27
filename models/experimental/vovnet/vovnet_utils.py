# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Union, Type, Tuple

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)

from tt_lib.fallback_ops import fallback_ops


def create_act_layer(name: Union[nn.Module, str], inplace=None, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        # recover if act layer doesn't have inplace arg
        return act_layer(**kwargs)


def get_act_layer(name: Union[Type[nn.Module], str] = "relu"):
    """Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    return nn.relu


def create_act(act_layer, act_kwargs=None, inplace=False, apply_act=True):
    act_layer = get_act_layer(act_layer)  # string -> nn.Module
    act_kwargs = act_kwargs or {}
    if act_layer is not None and apply_act:
        if inplace:
            act_kwargs["inplace"] = inplace
        act = act_layer(**act_kwargs)
    else:
        act = nn.Identity()
    return act


def adaptive_pool_feat_mult(pool_type="avg"):
    if pool_type.endswith("catavgmax"):
        return 2
    else:
        return 1


def create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert "num_experts" not in kwargs  # MixNet + CondConv combo not supported currently
        if "groups" in kwargs:
            groups = kwargs.pop("groups")
            if groups == in_channels:
                kwargs["depthwise"] = True
            else:
                assert groups == 1
    else:
        depthwise = kwargs.pop("depthwise", False)
        # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop("groups", 1)
        m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if not is_dynamic:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_batchnorm(out_ch, state_dict, base_address: str, device=None):
    weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.weight"], device, put_on_device=False)
    bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.bias"], device, put_on_device=False)
    running_mean = torch_to_tt_tensor_rm(state_dict[f"{base_address}.running_mean"], device, put_on_device=False)
    running_variance = torch_to_tt_tensor_rm(state_dict[f"{base_address}.running_var"], device, put_on_device=False)
    num_batches_tracked = torch_to_tt_tensor_rm(
        state_dict[f"{base_address}.num_batches_tracked"], device, put_on_device=False
    )
    norm = fallback_ops.BatchNorm2d(
        weight,
        bias,
        running_mean,
        running_variance,
        num_batches_tracked,
        out_ch,
    )

    return norm
