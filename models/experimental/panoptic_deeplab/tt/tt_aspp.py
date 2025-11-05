# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from math import floor

import ttnn
from loguru import logger

from models.tt_cnn.tt.builder import TtConv2d
from models.common.lightweightmodule import LightweightModule


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
            logger.info(f"🔷 Executing conv: aspp.convs.{i}")
            logger.info(f"  Input x: shape={x.shape}, dtype={x.dtype}, layout={x.layout}, is_sharded={x.is_sharded()}")

            # Branches 1-3 use channel slicing and need manual ReLU
            # Branch 0 uses no slicing and can use fused ReLU
            needs_manual_relu = i in [1, 2, 3]

            # Special handling for branch 3: needs flattened format for bfloat8_b dilated convolutions
            # Block float dtypes require flattened format (1, 1, nhw, C) for dilated convolutions
            if i == 3 and x.dtype == ttnn.bfloat8_b:
                logger.info(f"  Branch {i}: Flattening input for bfloat8_b dilated convolution")
                x_for_conv = ttnn.reshape(x, (1, 1, H * W, C))
                logger.info(f"    Flattened: {x.shape} -> {x_for_conv.shape}")

                branch_out = conv(x_for_conv)
                branch_out = ttnn.reshape(branch_out, (N, H, W, branch_out.shape[3]))
                if needs_manual_relu:
                    branch_out = ttnn.relu(branch_out)
            else:
                branch_out = conv(x)

                # Reshape logic for aspp.convs.0 that now outputs flattened format [1,1,NHW,C] -> [N,H,W,C]
                if i == 0:
                    if branch_out.shape[1] == 1 and branch_out.shape[2] > 1:
                        # Flattened format: [1, 1, NHW, C] -> [1, H, W, C]
                        NHW = branch_out.shape[2]
                        C_out = branch_out.shape[3]  # Use different variable name to avoid overwriting C
                        # W = 2*H, so NHW = N*H*W = 1*H*2*H = 2*H^2, therefore H = sqrt(NHW/2)
                        H_out = int((NHW // 2) ** 0.5)
                        W_out = 2 * H_out
                        if H_out * W_out == NHW:
                            branch_out = ttnn.reshape(branch_out, (N, H_out, W_out, C_out))
                            logger.debug(
                                f"aspp.convs.0 reshaped output from [1, 1, {NHW}, {C_out}] to [{N}, {H_out}, {W_out}, {C_out}]"
                            )
                elif branch_out.shape[1] == 1 and branch_out.shape[2] == H * W:
                    branch_out = ttnn.reshape(branch_out, (N, H, W, branch_out.shape[3]))

                if needs_manual_relu:
                    branch_out = ttnn.relu(branch_out)

            res.append(branch_out)

        # Global average pooling branch
        # If input is bfloat8_b, it needs to be flattened for pooling
        x_for_pooling = x
        if x.dtype == ttnn.bfloat8_b and x.shape[1] == H and x.shape[2] == W:
            logger.info(f"  Flattening input for global pooling: {x.shape} -> (1, 1, {H*W}, {C})")
            x_for_pooling = ttnn.reshape(x, (1, 1, H * W, C))

        pooled = ttnn.avg_pool2d(
            input_tensor=x_for_pooling,
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
        logger.info(f"🔷 Executing conv: aspp.convs.4")
        pooled = self.pool_conv(pooled)
        # Reshape logic for aspp.convs.4 that now outputs flattened format [1,1,NHW,C] -> [N,H,W,C]
        if pooled.shape[1] == 1 and pooled.shape[2] > 1:
            # Flattened format: [1, 1, NHW, C] -> [1, H, W, C]
            NHW = pooled.shape[2]
            C_pooled = pooled.shape[3]  # Use different variable name to avoid overwriting C
            # W = 2*H, so NHW = N*H*W = 1*H*2*H = 2*H^2, therefore H = sqrt(NHW/2)
            H_out = int((NHW // 2) ** 0.5)
            W_out = 2 * H_out
            if H_out * W_out == NHW:
                pooled = ttnn.reshape(pooled, (N, H_out, W_out, C_pooled))
                logger.debug(
                    f"aspp.convs.4 reshaped output from [1, 1, {NHW}, {C_pooled}] to [{N}, {H_out}, {W_out}, {C_pooled}]"
                )

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
        # accuracy changes are negligible, so we use nearest for performance reasons
        upsample_mode = "nearest"
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
        logger.info(f"🔷 Executing conv: aspp.project")
        res = self.project_conv(res)
        # Reshape logic for aspp.project that now outputs flattened format [1,1,NHW,C] -> [N,H,W,C]
        if res.shape[1] == 1 and res.shape[2] > 1:
            # Flattened format: [1, 1, NHW, C] -> [1, H, W, C]
            NHW = res.shape[2]
            C_project = res.shape[3]  # Use different variable name to avoid overwriting C
            # W = 2*H, so NHW = N*H*W = 1*H*2*H = 2*H^2, therefore H = sqrt(NHW/2)
            H_out = int((NHW // 2) ** 0.5)
            W_out = 2 * H_out
            if H_out * W_out == NHW:
                res = ttnn.reshape(res, (N, H_out, W_out, C_project))
                logger.debug(
                    f"aspp.project reshaped output from [1, 1, {NHW}, {C_project}] to [{N}, {H_out}, {W_out}, {C_project}]"
                )
        logger.debug(f"TtASPP forward pass complete - output shape: {res.shape}")

        return res
