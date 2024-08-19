# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger
from tt_lib.fallback_ops import fallback_ops
from typing import Optional, Sequence, Tuple, Union

from models.utility_functions import (
    torch2tt_tensor,
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)


class TtEfficientnetConv2d(torch.nn.Module):
    """
    Configurable block used for Convolution2d blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        state_dict,
        base_address,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        conv_on_device=False,
    ):
        super().__init__()

        self.conv_weight = state_dict[f"{base_address}.weight"]
        bias_key = f"{base_address}.bias"

        if bias_key in state_dict:
            self.conv_bias = state_dict[bias_key]
        else:
            self.conv_bias = None

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))

        # self.conv =
        #     torch.nn.Conv2d(
        #         in_channels,
        #         out_channels,
        #         kernel_size,
        #         stride,
        #         padding,
        #         dilation=dilation,
        #         groups=groups,
        #         bias=bias,)

        self.device = device
        self.conv_on_device = conv_on_device

        # conv_params = [out_channels, in_channels, kernel_size, kernel_size, stride, stride, padding, padding, dilation, groups]
        self.conv_params = [
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding,
            dilation,
            groups,
        ]

        if self.conv_on_device and is_conv_supported_on_device(self.conv_params):
            logger.debug(f"Using TtConv for params {self.conv_params}")

            self.conv = run_conv_on_device_wrapper(
                self.conv_weight.reshape(-1).tolist(),
                self.conv_params,
                self.device,
                conv_bias=None,
            )

        else:
            self.conv_on_device = False
            logger.debug(f"Using fallback_ops.Conv2d for params {self.conv_params}")

            # self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
            self.conv = fallback_ops.Conv2d(
                weights=self.conv_weight,
                biases=self.conv_bias,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=self.conv_bias is not None,
            )

    def forward(self, x):
        return self.conv(x)


class TtEfficientnetConv2dNormActivation(torch.nn.Module):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (bool): True if to use activation (Silu), false othervise.
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        state_dict,
        conv_base_address,
        bn_base_address,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer_eps: float = 1e-05,
        norm_layer_momentum: float = 0.1,
        activation_layer: bool = True,
        dilation: Union[int, Tuple[int, ...]] = 1,
        conv_on_device=False,
        is_lite=False,
    ):
        super().__init__()

        self.conv2d = TtEfficientnetConv2d(
            state_dict=state_dict,
            base_address=conv_base_address,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            conv_on_device=conv_on_device,
        )

        bnorm_weights = state_dict[f"{bn_base_address}.weight"]
        bnrom_bias = state_dict[f"{bn_base_address}.bias"]
        running_mean = state_dict[f"{bn_base_address}.running_mean"]
        running_var = state_dict[f"{bn_base_address}.running_var"]

        bnorm_weights = torch2tt_tensor(bnorm_weights, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        bnrom_bias = torch2tt_tensor(bnrom_bias, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        running_mean = torch2tt_tensor(running_mean, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        running_var = torch2tt_tensor(running_var, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.bnorm = fallback_ops.BatchNorm2d(
            weights=bnorm_weights,
            biases=bnrom_bias,
            running_mean=running_mean,
            running_var=running_var,
            num_batches_tracked=state_dict[f"{bn_base_address}.num_batches_tracked"],
            num_features=out_channels,
            eps=norm_layer_eps,
            momentum=norm_layer_momentum,
        )

        self.bnorm.eval()
        self.activation_layer = activation_layer
        self.is_lite = is_lite

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bnorm(x)

        if self.activation_layer is True:
            if self.is_lite:
                # Lite variant has ReLU6 instead of silu
                x = ttnn.relu6(x)
            else:
                x = ttnn.silu(x)

        return x
