# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from math import floor

import ttnn
from typing import Any
from loguru import logger

from models.tt_cnn.tt.builder import TtConv2d
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
    """
    TTNN implementation of ASPP using TT CNN Builder API.

    ASPP (Atrous Spatial Pyramid Pooling) consists of:
    - Branch 0: 1x1 convolution
    - Branches 1-3: 3x3 convolutions with different dilation rates
    - Branch 4: Global average pooling + 1x1 conv
    - Project layer: 1x1 convolution to combine all branches
    """

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
            f"Initializing TtASPP with TT CNN Builder - in_channels: {in_channels}, out_channels: {out_channels}, dilations: {dilations}"
        )
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.dropout = dropout
        self.model_configs = model_configs
        use_bias = False

        self.device = device
        self.activation = ttnn.relu  # For manual ReLU on channel-sliced branches
        self.pool_kernel_size = pool_kernel_size

        self.conv_branches = []

        # Branch 0: 1x1 convolution (BatchNorm fused)
        conv0_params = parameters["convs"][0]
        conv0 = self._create_conv_layer(conv0_params, "aspp.convs.0")
        self.conv_branches.append(conv0)

        # Branches 1-3: 3x3 convolutions with dilations (BatchNorm fused)
        for i, dilation in enumerate(dilations):
            conv_idx = i + 1
            conv_params = parameters["convs"][conv_idx]
            conv = self._create_conv_layer(conv_params, f"aspp.convs.{conv_idx}")
            self.conv_branches.append(conv)

        # Branch 4: Global pooling + 1x1 conv (BatchNorm fused)
        pool_conv_params = parameters["convs"][4][1]  # pooling branch is convs[4], then [1] for the Conv2d part
        self.pool_conv = self._create_conv_layer(pool_conv_params, "aspp.convs.4")

        # Final Project convolution (BatchNorm fused)
        project_conv_params = parameters["project"]
        self.project_conv = self._create_conv_layer(project_conv_params, "aspp.project")

        logger.debug("TtASPP initialization complete")

    def _create_conv_layer(self, params, conv_path: str):
        """Helper to create conv layer using TT CNN Builder with config overrides"""
        # Get base Conv2dConfiguration from preprocessing
        if "conv_config" in params:
            base_config = params["conv_config"]
            logger.debug(f"Using Conv2dConfiguration from preprocessing for {conv_path}")
        else:
            logger.error(f"Conv2dConfiguration not found for {conv_path}")
            raise ValueError(f"Expected 'conv_config' in parameters for {conv_path}")

        # Apply model-specific overrides if model_configs is provided
        if self.model_configs is not None:
            final_config = self.model_configs.apply_conv_overrides(base_config, conv_path=conv_path)
            logger.debug(f"Applied config overrides for {conv_path}")
        else:
            final_config = base_config
            logger.debug(f"No model_configs for {conv_path}, using base config")

        # Create TtConv2d using TT CNN Builder
        return TtConv2d(final_config, self.device)

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

        # Process conv branches
        res = []
        for i, conv in enumerate(self.conv_branches):
            branch_out = conv(x)

            # Channel-sliced convolutions may output in flattened format [N, 1, H*W, C]
            # Reshape back to [N, H, W, C] and apply ReLU manually
            if branch_out.shape[1] == 1 and branch_out.shape[2] == H * W:
                branch_out = ttnn.reshape(branch_out, (N, H, W, branch_out.shape[3]))
                # Apply ReLU manually (fused ReLU doesn't work with channel slicing)
                branch_out = self.activation(branch_out)

            res.append(branch_out)

        # Global average pooling branch
        pooled = ttnn.avg_pool2d(
            input_tensor=x,
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

        # Move all branches to DRAM
        for i in range(len(res)):
            res[i] = ttnn.to_memory_config(res[i], ttnn.DRAM_MEMORY_CONFIG)
        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        # Process pooled branch with conv
        pooled = self.pool_conv(pooled)

        # Upsample pooled branch to match input spatial dimensions
        current_h, current_w = pooled.shape[1], pooled.shape[2]
        scale_factor = [H // current_h, W // current_w]

        # Convert to interleaved and ROW_MAJOR for upsample
        if pooled.is_sharded():
            pooled = ttnn.sharded_to_interleaved(pooled, ttnn.DRAM_MEMORY_CONFIG)
        else:
            pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        pooled = ttnn.to_layout(pooled, ttnn.ROW_MAJOR_LAYOUT)

        # Choose upsample mode: nearest for 1x1, bilinear otherwise
        upsample_mode = "nearest" if (current_h == 1 and current_w == 1) else "bilinear"
        pooled = ttnn.upsample(pooled, scale_factor=tuple(scale_factor), mode=upsample_mode)

        # Convert back to TILE_LAYOUT
        pooled = ttnn.to_layout(pooled, ttnn.TILE_LAYOUT)
        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        res.append(pooled)

        # Ensure all tensors have the same dtype before concatenation
        target_dtype = ttnn.bfloat8_b
        for i in range(len(res)):
            if res[i].dtype != target_dtype:
                res[i] = ttnn.typecast(res[i], target_dtype)

        # Concatenate all branches
        res = ttnn.concat(res, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtASPP concatenated branches, shape: {res.shape}")

        # Project to final output
        res = self.project_conv(res)
        logger.debug(f"TtASPP forward pass complete - output shape: {res.shape}")

        return res
