# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from math import floor
import torch.nn as nn
import torch

import ttnn

from models.experimental.panoptic_deeplab.tt.tt_conv2dWrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)
from models.experimental.panoptic_deeplab.tt.tt_upsample_wrapper import TtUpsample


def get_ttnn_activation(activation_name: str):
    """Returns a ttnn activation function."""
    if activation_name.lower() == "silu":
        return ttnn.silu
    elif activation_name.lower() == "relu":
        return ttnn.relu
    else:
        raise NotImplementedError(f"Activation '{activation_name}' not supported in ttnn.")


def get_ttnn_norm(norm_name: str, num_channels: int, device, norm_params: dict = None):
    """
    Returns a ttnn normalization function.

    Args:
        norm_name: Type of normalization ('BN'/'SyncBN', 'GN', 'LN', or '')
        num_channels: Number of channels to normalize
        device: TTNN device
        norm_params: Optional dictionary with pre-trained normalization parameters
                    Expected keys: 'weight', 'bias', 'running_mean', 'running_var'
    """
    if norm_name.lower() in ["bn", "syncbn", "batchnorm"]:
        # BatchNorm / SyncBN implementation
        if norm_params is not None:
            weight = ttnn.from_torch(
                norm_params.get("weight", torch.ones(num_channels, dtype=torch.bfloat16)).view(1, -1, 1, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            bias = ttnn.from_torch(
                norm_params.get("bias", torch.zeros(num_channels, dtype=torch.bfloat16)).view(1, -1, 1, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            running_mean = ttnn.from_torch(
                norm_params.get("running_mean", torch.zeros(num_channels, dtype=torch.bfloat16)).view(1, -1, 1, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            running_var = ttnn.from_torch(
                norm_params.get("running_var", torch.ones(num_channels, dtype=torch.bfloat16)).view(1, -1, 1, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
        else:
            weight = ttnn.ones((1, num_channels, 1, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            bias = ttnn.zeros((1, num_channels, 1, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            running_mean = ttnn.zeros(
                (1, num_channels, 1, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            running_var = ttnn.ones(
                (1, num_channels, 1, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

        def batch_norm_fn(x):
            x_nchw = ttnn.permute(x, (0, 3, 1, 2))

            x_normed = ttnn.batch_norm(
                x_nchw,
                running_mean=running_mean,
                running_var=running_var,
                weight=weight,
                bias=bias,
                eps=1e-5,
                training=False,
            )

            x_nhwc = ttnn.permute(x_normed, (0, 2, 3, 1))

            return x_nhwc

        return batch_norm_fn
    elif norm_name.lower() == "gn":
        num_groups = 32
        weight = ttnn.ones((1, 1, 1, num_channels), device=device, layout=ttnn.TILE_LAYOUT)
        bias = ttnn.zeros((1, 1, 1, num_channels), device=device, layout=ttnn.TILE_LAYOUT)
        return lambda x: ttnn.group_norm(x, num_groups=num_groups, weight=weight, bias=bias)
    elif norm_name.lower() == "ln":
        weight = ttnn.ones((1, 1, 1, num_channels), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        bias = ttnn.zeros((1, 1, 1, num_channels), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        return lambda x: ttnn.layer_norm(x, weight=weight, bias=bias)
    elif norm_name == "":
        return lambda x: x  # No-op
    else:
        raise NotImplementedError(
            f"Normalization '{norm_name}' not supported. Supported: 'BN'/'SyncBN', 'GN', 'LN', or '' (none)."
        )


class TtASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        device: ttnn.MeshDevice,
        *,
        norm,
        activation,
        dropout: float = 0.0,
        pool_kernel_size,
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
    ):
        super(TtASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.dropout = dropout
        use_bias = False
        self.conv_branches = []

        self.activation = get_ttnn_activation(activation)
        self.device = device
        self.pool_kernel_size = pool_kernel_size

        self.shared_weight_tensor_kernel1 = shared_weight_tensor_kernel1
        self.shared_weight_tensor_kernel3 = shared_weight_tensor_kernel3
        self.shared_weight_tensor_kernel1_output5 = shared_weight_tensor_kernel1_output5

        def create_ttconv2d(
            out_channels, kernel_size, stride, padding, channel_slice_num, dilation=1, bias=False, isProjectConv=False
        ):
            if kernel_size == 1 and isProjectConv:
                weight = self.shared_weight_tensor_kernel1_output5
            elif kernel_size == 3:
                weight = self.shared_weight_tensor_kernel3
            else:
                weight = self.shared_weight_tensor_kernel1
            param_dict = {
                "weight": weight,
                "dilation": (dilation, dilation),
            }
            if bias:
                param_dict["bias"] = torch.empty(1, 1, 1, out_channels, dtype=torch.bfloat16)

            # Create slice configuration
            slice_config = None
            if channel_slice_num > 1:
                slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=channel_slice_num)

            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device, slice_config=slice_config)

            return TtConv2d(parameters, stride=stride, padding=padding)

        # 1x1 conv
        conv = create_ttconv2d(
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            channel_slice_num=1,
            bias=use_bias,
        )
        norm_func = get_ttnn_norm(norm, out_channels, device=self.device, norm_params=None)
        self.conv_branches.append((conv, norm_func))

        channel_slices = [2, 4, 8]
        i = 0
        # Dilations convs
        for dilation in dilations:
            channel_slice_num = channel_slices[i]
            i += 1
            conv = create_ttconv2d(
                out_channels=out_channels,
                kernel_size=3,
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=dilation,
                channel_slice_num=channel_slice_num,
                bias=use_bias,
            )
            norm_func = get_ttnn_norm(norm, out_channels, device=self.device, norm_params=None)
            self.conv_branches.append((conv, norm_func))

        # Global pooling conv
        self.pool_conv = create_ttconv2d(
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            channel_slice_num=1,
            bias=False,
        )

        # Project conv to concatenate all branches
        self.project_conv = create_ttconv2d(
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            channel_slice_num=1,
            bias=use_bias,
            isProjectConv=True,
        )

        self.project_norm = get_ttnn_norm(norm, out_channels, device=self.device, norm_params=None)

        # Initialize upsample wrapper for global pooling branch
        self.pool_upsample = TtUpsample.create(device=device, scale_factor=(1, 1), mode="bilinear")

    def forward(self, x):
        input_shape = x.shape
        N = x.shape[0]  # Batch size
        H = x.shape[1]  # Height
        W = x.shape[2]  # Width
        C = x.shape[3]  # Channels

        if H % self.pool_kernel_size[0] or W % self.pool_kernel_size[1]:
            raise ValueError(
                "`pool_kernel_size` must be divisible by the shape of inputs. "
                "Input size: {} `pool_kernel_size`: {}".format(input_shape, self.pool_kernel_size)
            )
        res = []
        for conv, norm in self.conv_branches:
            branch_out = conv(x)
            branch_out = norm(branch_out)
            branch_out = self.activation(branch_out)
            res.append(branch_out)
        input_shape = (1, 1, N * H * W, C)
        ttnn_reshape = x.reshape(input_shape)

        pooled = ttnn.avg_pool2d(
            input_tensor=ttnn_reshape,
            batch_size=N,
            input_h=H,
            input_w=W,
            channels=C,
            kernel_size=self.pool_kernel_size,
            stride=[1, 1],
            padding=[0, 0],
            ceil_mode=False,
        )

        output_h = floor(H + 0 - self.pool_kernel_size[0]) + 1
        output_w = floor(W + 0 - self.pool_kernel_size[1]) + 1

        pooled = ttnn.reshape(pooled, (N, output_h, output_w, C))

        for i in range(len(res)):
            res[i] = ttnn.to_memory_config(res[i], ttnn.DRAM_MEMORY_CONFIG)
        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        pooled = self.pool_conv(pooled)
        pooled = self.activation(pooled)

        current_h, current_w = pooled.shape[1], pooled.shape[2]
        scale_factor = [H // current_h, W // current_w]

        # Set appropriate mode and scale factor for upsample
        if pooled.shape[1] == 1 and pooled.shape[2] == 1:
            self.pool_upsample._mode = "nearest"
        else:
            self.pool_upsample._mode = "bilinear"
        self.pool_upsample._scale_factor = scale_factor
        pooled = self.pool_upsample(pooled)
        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        res.append(pooled)

        res = ttnn.concat(res, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        res = self.project_conv(res)
        res = self.project_norm(res)
        res = self.activation(res)
        return res
