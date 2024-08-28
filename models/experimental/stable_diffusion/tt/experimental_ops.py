# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import ttnn
from contextlib import AbstractContextManager
from loguru import logger
from functools import wraps
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)


class UseDeviceConv:
    READY = True


def disable_conv(fn):
    @wraps(fn)
    def __wrapper__(*args, **kwargs):
        with DisableDeviceConv(use_conv=False) as _:
            values = fn(*args, **kwargs)
        return values

    return __wrapper__


class DisableDeviceConv(AbstractContextManager):
    """useful for testing"""

    def __init__(self, use_conv=False):
        self.state = [UseDeviceConv.READY]
        UseDeviceConv.READY = use_conv
        logger.debug("Disabled Device Conv operators.")

    def __exit__(self, exc_type, exc_value, traceback):
        UseDeviceConv.READY = self.state[0]
        logger.debug("Restored Device Conv operators.")
        del self.state


def parse_conv2d_interface(conv_weight=None, conv_bias=None, in_channels=-1, out_channels=-1, **kwargs):
    # conv_weight, conv_bias, in_channels, out_channels, **kwargs
    # conv_weight, conv_bias, in_channels, out_channels, kernel_size, stride, padding
    if "biases" in kwargs:
        bias = kwargs["biases"]
    elif "bias" in kwargs:
        bias = kwargs["bias"]
    else:
        bias = conv_bias
    if "weights" in kwargs:
        weights = kwargs["weights"]
    else:
        weights = conv_weight
    kernel_size = kwargs["kernel_size"]  # required
    stride = kwargs.get("stride", (1, 1))
    padding = kwargs.get("padding", (0, 0))
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    if isinstance(kernel_size, (list, tuple)):
        kernel_x, kernel_y = kernel_size
    else:
        kernel_x = kernel_size
        kernel_y = kernel_size

    if isinstance(padding, (list, tuple)):
        padding_x, padding_y = padding
    else:
        padding_x = padding
        padding_y = padding

    if isinstance(stride, (list, tuple)):
        stride_x, stride_y = stride
    else:
        stride_x = stride
        stride_y = stride

    # in run_conv_on_device_wrapper format
    params = [
        out_channels,
        in_channels,
        kernel_x,
        kernel_y,
        stride_x,
        stride_y,
        padding_x,
        padding_y,
        dilation,
        groups,
    ]
    return weights, bias, params


def Conv2d(*args, **kwargs):
    if UseDeviceConv.READY:
        conv1_weight, conv1_bias, conv1_params = parse_conv2d_interface(*args, **kwargs)
        device = ttnn.GetDefaultDevice()
        return run_conv_on_device_wrapper(
            conv1_weight.reshape(-1).tolist(),
            conv1_params,
            device,
            conv1_bias.tolist() if conv1_bias is not None else None,
            channel_transpose=True,
        )

    return fallback_ops.Conv2d(*args, **kwargs)


def concat(tensors, dim=0):
    device = ttnn.GetDefaultDevice()
    new_tensors = []
    for t in tensors:
        if torch.is_tensor(t):
            t = ttnn.Tensor(t, ttnn.bfloat16).to(device)
        assert isinstance(t, ttnn.Tensor)
        if t.storage_type() != ttnn.StorageType.DEVICE:
            t = t.to(device)
        new_tensors.append(t)

    return ttnn.concat(new_tensors, dim)
