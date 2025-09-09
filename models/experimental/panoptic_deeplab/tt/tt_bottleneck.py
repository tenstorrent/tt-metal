# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)


class TtBottleneck(nn.Module):
    """
    TTNN implementation of BottleneckBlock.

    Based on the model structure, BottleneckBlock contains:
    - conv1: 1x1 conv (channel reduction)
    - conv2: 3x3 conv (spatial convolution, potentially with stride/dilation)
    - conv3: 1x1 conv (channel expansion)
    - shortcut: optional 1x1 conv for residual connection when input/output dimensions differ
    Each with SyncBatchNorm and ReLU activation, except the final output uses residual addition + ReLU.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        has_shortcut: bool = False,
        stride: int = 1,
        dilation: int = 1,
        shortcut_stride: int = 1,
        block_id: str = "unknown",
    ):
        super().__init__()
        self.device = device
        self.has_shortcut = has_shortcut
        self.block_id = block_id

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
        )

        # For conv2, use architecture parameters and update dilation
        conv2_tt_params = TtConv2dParameters.from_preprocessed_parameters(conv2_params, device=device, dtype=dtype)
        conv2_tt_params.dilation = conv2_dilation
        self.conv2 = TtConv2d(conv2_tt_params, stride=conv2_stride, padding=conv2_padding)

        self.conv3 = TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(conv3_params, device=device, dtype=dtype),
            stride=(1, 1),
            padding=(0, 0),
        )

        # Initialize shortcut if needed
        if has_shortcut:
            shortcut_params = parameters["shortcut"]
            shortcut_stride_tuple = (shortcut_stride, shortcut_stride)

            # Apply width slicing for res3 blocks
            shortcut_slice_config = None
            if block_id.startswith("res3"):
                shortcut_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=2)

            self.shortcut = TtConv2d(
                TtConv2dParameters.from_preprocessed_parameters(
                    shortcut_params, device=device, dtype=dtype, slice_config=shortcut_slice_config
                ),
                stride=shortcut_stride_tuple,
                padding=(0, 0),
            )

        # Extract normalization parameters from preprocessed structure
        conv1_norm_params = conv1_params.get("norm", None)
        conv2_norm_params = conv2_params.get("norm", None)
        conv3_norm_params = conv3_params.get("norm", None)

        # Normalization parameters are already TTNN tensors from preprocessing
        # Conv1 normalization
        if conv1_norm_params:
            self.conv1_norm_weight = conv1_norm_params["weight"]
            self.conv1_norm_bias = conv1_norm_params["bias"]
            self.conv1_norm_running_mean = conv1_norm_params["running_mean"]
            self.conv1_norm_running_var = conv1_norm_params["running_var"]
        else:
            self.conv1_norm_weight = None
            self.conv1_norm_bias = None
            self.conv1_norm_running_mean = None
            self.conv1_norm_running_var = None

        # Conv2 normalization
        if conv2_norm_params:
            self.conv2_norm_weight = conv2_norm_params["weight"]
            self.conv2_norm_bias = conv2_norm_params["bias"]
            self.conv2_norm_running_mean = conv2_norm_params["running_mean"]
            self.conv2_norm_running_var = conv2_norm_params["running_var"]
        else:
            self.conv2_norm_weight = None
            self.conv2_norm_bias = None
            self.conv2_norm_running_mean = None
            self.conv2_norm_running_var = None

        # Conv3 normalization
        if conv3_norm_params:
            self.conv3_norm_weight = conv3_norm_params["weight"]
            self.conv3_norm_bias = conv3_norm_params["bias"]
            self.conv3_norm_running_mean = conv3_norm_params["running_mean"]
            self.conv3_norm_running_var = conv3_norm_params["running_var"]
        else:
            self.conv3_norm_weight = None
            self.conv3_norm_bias = None
            self.conv3_norm_running_mean = None
            self.conv3_norm_running_var = None

        # Shortcut normalization (if exists)
        if has_shortcut:
            shortcut_norm_params = shortcut_params.get("norm", None)
            if shortcut_norm_params:
                self.shortcut_norm_weight = shortcut_norm_params["weight"]
                self.shortcut_norm_bias = shortcut_norm_params["bias"]
                self.shortcut_norm_running_mean = shortcut_norm_params["running_mean"]
                self.shortcut_norm_running_var = shortcut_norm_params["running_var"]
            else:
                self.shortcut_norm_weight = None
                self.shortcut_norm_bias = None
                self.shortcut_norm_running_mean = None
                self.shortcut_norm_running_var = None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logger.debug(f"TtBottleneck {self.block_id} forward pass starting, input shape: {x.shape}")

        # Store input for residual connection
        identity = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # workaround for conv tilize issue with non-height shard
        if identity.spec.layout != ttnn.TILE_LAYOUT:
            identity = ttnn.tilize(identity)
        # ttnn.deallocate(x)

        # Process shortcut if needed
        if self.has_shortcut:
            logger.debug(f"TtBottleneck {self.block_id} processing shortcut convolution")
            identity = self.shortcut(identity)
            identity = ttnn.to_memory_config(identity, ttnn.DRAM_MEMORY_CONFIG)
            # Convert NHWC to NCHW for batch_norm
            identity = ttnn.permute(identity, (0, 3, 1, 2))
            logger.debug(f"TtBottleneck {self.block_id} applying shortcut batch norm")
            identity = ttnn.batch_norm(
                identity,
                running_mean=self.shortcut_norm_running_mean,
                running_var=self.shortcut_norm_running_var,
                weight=self.shortcut_norm_weight,
                bias=self.shortcut_norm_bias,
                eps=1e-05,
                training=False,
            )
            # Convert back to NHWC
            identity = ttnn.permute(identity, (0, 2, 3, 1))
            logger.debug(f"TtBottleneck {self.block_id} shortcut processing complete, shape: {identity.shape}")

        # Main path: Conv1 + BatchNorm + ReLU
        logger.debug(f"TtBottleneck {self.block_id} processing conv1 (1x1 reduction)")
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        out = self.conv1(x)
        # Convert NHWC to NCHW for batch_norm
        out = ttnn.permute(out, (0, 3, 1, 2))
        logger.debug(f"TtBottleneck {self.block_id} applying conv1 batch norm")
        out = ttnn.batch_norm(
            out,
            running_mean=self.conv1_norm_running_mean,
            running_var=self.conv1_norm_running_var,
            weight=self.conv1_norm_weight,
            bias=self.conv1_norm_bias,
            eps=1e-05,
            training=False,
        )
        # Convert back to NHWC
        out = ttnn.permute(out, (0, 2, 3, 1))
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtBottleneck {self.block_id} conv1 complete, output shape: {out.shape}")

        # Conv2 + BatchNorm + ReLU
        logger.debug(f"TtBottleneck {self.block_id} processing conv2 (3x3 spatial)")
        out = self.conv2(out)
        # Convert NHWC to NCHW for batch_norm
        # out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.permute(out, (0, 3, 1, 2))
        logger.debug(f"TtBottleneck {self.block_id} applying conv2 batch norm")
        out = ttnn.batch_norm(
            out,
            running_mean=self.conv2_norm_running_mean,
            running_var=self.conv2_norm_running_var,
            weight=self.conv2_norm_weight,
            bias=self.conv2_norm_bias,
            eps=1e-05,
            training=False,
        )
        # Convert back to NHWC
        out = ttnn.permute(out, (0, 2, 3, 1))
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtBottleneck {self.block_id} conv2 complete, output shape: {out.shape}")

        # Conv3 + BatchNorm (no ReLU yet)
        logger.debug(f"TtBottleneck {self.block_id} processing conv3 (1x1 expansion)")
        out = self.conv3(out)
        logger.debug(f"TtBottleneck {self.block_id} conv3 convolution complete, output shape: {out.shape}")
        # Convert NHWC to NCHW for batch_norm
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.permute(out, (0, 3, 1, 2))
        logger.debug(f"TtBottleneck {self.block_id} applying conv3 batch norm")
        out = ttnn.batch_norm(
            out,
            running_mean=self.conv3_norm_running_mean,
            running_var=self.conv3_norm_running_var,
            weight=self.conv3_norm_weight,
            bias=self.conv3_norm_bias,
            eps=1e-05,
            training=False,
        )
        # Convert back to NHWC
        out = ttnn.permute(out, (0, 2, 3, 1))
        logger.debug(f"TtBottleneck {self.block_id} conv3 complete, output shape: {out.shape}")

        # Residual connection + ReLU
        logger.debug(f"TtBottleneck {self.block_id} adding residual connection and applying final ReLU")
        if self.has_shortcut or identity.shape == out.shape:
            out = ttnn.add(out, identity)
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        logger.debug(f"TtBottleneck {self.block_id} forward pass complete, final output shape: {out.shape}")
        return out
