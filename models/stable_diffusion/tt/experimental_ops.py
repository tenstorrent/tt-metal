import torch
import tt_lib as ttl

from tt_lib.fallback_ops import fallback_ops
from models.conv_on_device_utils import (
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)

DEVICE_CONV2D_READY = True  # or goto fallback
DEVICE_CONCAT_READY = True


def parse_conv2d_interface(
    conv_weight=None, conv_bias=None, in_channels=-1, out_channels=-1, **kwargs
):
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
    if DEVICE_CONV2D_READY:
        conv1_weight, conv1_bias, conv1_params = parse_conv2d_interface(*args, **kwargs)
        device = ttl.device.GetDefaultDevice()
        return run_conv_on_device_wrapper(
            conv1_weight.reshape(-1).tolist(),
            conv1_params,
            device,
            conv1_bias.tolist() if conv1_bias is not None else None,
            channel_transpose=True,
        )

    return fallback_ops.Conv2d(*args, **kwargs)


def concat(*args, **kwargs):
    if DEVICE_CONCAT_READY:
        dim = kwargs.get("dim", 0)
        return ttl.tensor.concat(*args, dim)
    return fallback_ops.concat(*args, **kwargs)
