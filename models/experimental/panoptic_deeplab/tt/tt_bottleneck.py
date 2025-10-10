# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)
from models.common.lightweightmodule import LightweightModule


class TtBottleneck(LightweightModule):
    """
    TTNN implementation of BottleneckBlock with fused Conv+BatchNorm.

    Based on the model structure, BottleneckBlock contains:
    - conv1: 1x1 conv (channel reduction) with bias
    - conv2: 3x3 conv (spatial convolution, potentially with stride/dilation) with bias
    - conv3: 1x1 conv (channel expansion) with bias
    - shortcut: optional 1x1 conv for residual connection when input/output dimensions differ
    Each with ReLU activation. BatchNorm operations are fused into the Conv weights and biases.
    The final output uses residual addition + ReLU.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
        has_shortcut: bool = False,
        stride: int = 1,
        dilation: int = 1,
        shortcut_stride: int = 1,
        block_id: str = "unknown",
        model_configs=None,
    ):
        super().__init__()
        self.device = device
        self.has_shortcut = has_shortcut
        self.block_id = block_id
        self.model_configs = model_configs

        # Extract parameters for conv1, conv2, conv3 from preprocessed structure
        conv1_params = parameters["conv1"]
        conv2_params = parameters["conv2"]
        conv3_params = parameters["conv3"]

        # Use passed stride and dilation parameters
        conv2_stride = (stride, stride)
        conv2_dilation = (dilation, dilation)
        conv2_padding = (dilation, dilation)  # Padding should match dilation for 3x3 conv

        # Initialize conv layers using preprocessed parameters
        self.conv1 = TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(conv1_params, device=device, dtype=dtype),
            stride=(1, 1),
            padding=(0, 0),
            conv_path=f"{block_id}.conv1",
            model_configs=model_configs,
            dtype=dtype,
        )

        # For conv2, use architecture parameters and update dilation
        conv2_tt_params = TtConv2dParameters.from_preprocessed_parameters(conv2_params, device=device, dtype=dtype)
        conv2_tt_params.dilation = conv2_dilation
        self.conv2 = TtConv2d(
            conv2_tt_params,
            stride=conv2_stride,
            padding=conv2_padding,
            conv_path=f"{block_id}.conv2",
            model_configs=model_configs,
            dtype=dtype,
        )

        self.conv3 = TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(conv3_params, device=device, dtype=dtype),
            stride=(1, 1),
            padding=(0, 0),
            conv_path=f"{block_id}.conv3",
            model_configs=model_configs,
            dtype=dtype,
        )

        # Initialize shortcut if needed
        if has_shortcut:
            shortcut_params = parameters["shortcut"]
            shortcut_stride_tuple = (shortcut_stride, shortcut_stride)

            # Apply width slicing for res3 blocks using model_configs
            shortcut_slice_config = None
            if block_id.startswith("res3"):
                if self.model_configs is not None:
                    res3_slice_config = self.model_configs.get_slice_config(f"{block_id}.shortcut")
                    if res3_slice_config["mode"] == "width":
                        shortcut_slice_config = SliceConfig(
                            mode=SliceMode.WIDTH, num_slices=res3_slice_config["num_slices"]
                        )
                else:
                    logger.warning(
                        f"FALLBACK BOTTLENECK SLICE CONFIG: Using default width slicing with num_slices=2 for {block_id}.shortcut instead of model_configs"
                    )
                    shortcut_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=2)

            self.shortcut = TtConv2d(
                TtConv2dParameters.from_preprocessed_parameters(
                    shortcut_params, device=device, dtype=dtype, slice_config=shortcut_slice_config
                ),
                stride=shortcut_stride_tuple,
                padding=(0, 0),
                conv_path=f"{block_id}.shortcut",
                model_configs=model_configs,
                dtype=dtype,
            )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logger.debug(f"TtBottleneck {self.block_id} forward pass starting, input shape: {x.shape}")

        # Store input for residual connection
        identity = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # workaround for conv tilize issue with non-height shard
        if identity.spec.layout != ttnn.TILE_LAYOUT:
            identity = ttnn.tilize(identity)

        # Process shortcut if needed (BatchNorm is now fused into shortcut Conv)
        if self.has_shortcut:
            logger.debug(f"TtBottleneck {self.block_id} processing shortcut convolution")
            identity = self.shortcut(identity)
            logger.debug(f"TtBottleneck {self.block_id} shortcut processing complete, shape: {identity.shape}")

        # Main path: Conv1 + ReLU (BatchNorm fused into Conv1)
        logger.debug(f"TtBottleneck {self.block_id} processing conv1 (1x1 reduction)")
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        out = self.conv1(x)
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtBottleneck {self.block_id} conv1 complete, output shape: {out.shape}")

        # Conv2 + ReLU (BatchNorm fused into Conv2)
        logger.debug(f"TtBottleneck {self.block_id} processing conv2 (3x3 spatial)")
        out = self.conv2(out)
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtBottleneck {self.block_id} conv2 complete, output shape: {out.shape}")

        # Conv3 (no ReLU yet, BatchNorm fused into Conv3)
        logger.debug(f"TtBottleneck {self.block_id} processing conv3 (1x1 expansion)")
        out = self.conv3(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtBottleneck {self.block_id} conv3 complete, output shape: {out.shape}")

        # Residual connection + ReLU
        logger.debug(f"TtBottleneck {self.block_id} adding residual connection and applying final ReLU")
        if self.has_shortcut or identity.shape == out.shape:
            out = ttnn.add(out, identity)
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        logger.debug(f"TtBottleneck {self.block_id} forward pass complete, final output shape: {out.shape}")
        return out
