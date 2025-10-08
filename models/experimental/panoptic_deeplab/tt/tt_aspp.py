# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from math import floor

import ttnn
from typing import Any
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import (
    TtConv2d,
    TtConv2dParameters,
)
from models.experimental.panoptic_deeplab.tt.tt_upsample_wrapper import TtUpsample
from models.common.lightweightmodule import LightweightModule


def get_ttnn_activation(activation_name: str):
    """Returns a ttnn activation function."""
    if activation_name.lower() == "silu":
        return ttnn.silu
    elif activation_name.lower() == "relu":
        return ttnn.relu
    else:
        raise NotImplementedError(f"Activation '{activation_name}' not supported in ttnn.")


def get_ttnn_norm(norm_name: str, num_channels: int, device, norm_params: Any = None):
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
            weight = norm_params.weight
            bias = norm_params.bias
            running_mean = norm_params.running_mean
            running_var = norm_params.running_var
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


class TtASPP(LightweightModule):
    def __init__(
        self,
        parameters,
        device: ttnn.Device,
        *,
        in_channels: int,
        out_channels: int,
        dilations,
        norm: str,
        activation: str,
        dropout: float = 0.0,
        pool_kernel_size,
        model_configs=None,
    ):
        super(TtASPP, self).__init__()
        logger.debug(
            f"Initializing TtASPP with in_channels: {in_channels}, out_channels: {out_channels}, dilations: {dilations}"
        )
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.dropout = dropout
        self.model_configs = model_configs
        use_bias = False

        self.activation = get_ttnn_activation(activation)
        self.device = device
        self.pool_kernel_size = pool_kernel_size

        self.conv_branches = []

        # Branch 1: 1x1 convolution (BatchNorm fused into Conv)
        conv0_path = parameters["convs"][0]
        conv0_bias = conv0_path.get("bias", None)

        conv0_params = TtConv2dParameters(
            weight=conv0_path["weight"],
            bias=conv0_bias,
            device=self.device,
        )
        conv0 = TtConv2d.create(
            conv0_params, stride=(1, 1), padding=(0, 0), conv_path="aspp.convs.0", model_configs=self.model_configs
        )
        self.conv_branches.append(conv0)

        # Branches 2, 3, 4: 3x3 convolutions with dilations (BatchNorm fused)
        # Default channel slices (fallback if model_configs not available)
        default_channel_slices = [2, 4, 8]

        for i, dilation in enumerate(dilations):
            conv_idx = i + 1
            conv_params_path = parameters["convs"][conv_idx]

            conv_bias = conv_params_path.get("bias", None)
            conv_params = TtConv2dParameters(
                weight=conv_params_path["weight"], bias=conv_bias, device=self.device, dilation=(dilation, dilation)
            )

            # Get slice config from model_configs or use default
            if self.model_configs is not None:
                aspp_slice_config = self.model_configs.get_aspp_slice_config(conv_idx)
                num_slices = aspp_slice_config["num_slices"]
            else:
                logger.warning(
                    f"FALLBACK ASPP SLICE CONFIG: Using default channel slicing with num_slices={default_channel_slices[i]} for aspp.convs.{conv_idx} instead of model_configs"
                )
                num_slices = default_channel_slices[i]

            conv = TtConv2d.create_with_channel_slicing(
                conv_params,
                stride=(1, 1),
                padding=(dilation, dilation),
                num_slices=num_slices,
                conv_path=f"aspp.convs.{i+1}",
                model_configs=self.model_configs,
            )
            self.conv_branches.append(conv)

        # Branch 5: Global pooling (BatchNorm fused into Conv)
        pool_conv_path = parameters["convs"][4][1]  # pooling branch is convs[4], then [1] for the Conv2d part
        pool_conv_bias = pool_conv_path.get("bias", None)
        pool_conv_params = TtConv2dParameters(
            weight=pool_conv_path["weight"],
            bias=pool_conv_bias,
            device=self.device,
        )
        self.pool_conv = TtConv2d.create(
            pool_conv_params, stride=(1, 1), padding=(0, 0), conv_path="aspp.convs.4", model_configs=self.model_configs
        )

        # Final Project convolution (BatchNorm fused into Conv)
        project_conv_path = parameters["project"]
        project_conv_bias = project_conv_path.get("bias", None)
        project_conv_params = TtConv2dParameters(
            weight=project_conv_path["weight"],
            bias=project_conv_bias,
            device=self.device,
        )
        self.project_conv = TtConv2d.create(
            project_conv_params,
            stride=(1, 1),
            padding=(0, 0),
            conv_path="aspp.project",
            model_configs=self.model_configs,
        )

        self.pool_upsample = TtUpsample.create(device=device, scale_factor=(1, 1), mode="bilinear")

        logger.debug("TtASPP initialization complete")

    def forward(self, x):
        input_shape = x.shape
        N = x.shape[0]  # Batch size
        H = x.shape[1]  # Height
        W = x.shape[2]  # Width
        C = x.shape[3]  # Channels

        logger.debug(f"TtASPP forward pass - input shape: {input_shape}, pool_kernel_size: {self.pool_kernel_size}")

        if H % self.pool_kernel_size[0] or W % self.pool_kernel_size[1]:
            raise ValueError(
                "`pool_kernel_size` must be divisible by the shape of inputs. "
                "Input size: {} `pool_kernel_size`: {}".format(input_shape, self.pool_kernel_size)
            )
        res = []
        for conv in self.conv_branches:
            branch_out = conv(x)
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

        # Ensure all tensors have the same dtype before concatenation
        target_dtype = ttnn.bfloat8_b
        for i in range(len(res)):
            if res[i].dtype != target_dtype:
                res[i] = ttnn.typecast(res[i], target_dtype)

        res = ttnn.concat(res, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtASPP concatenated branches, shape: {res.shape}")

        res = self.project_conv(res)
        res = self.activation(res)
        logger.debug(f"TtASPP forward pass complete - output shape: {res.shape}")
        return res
