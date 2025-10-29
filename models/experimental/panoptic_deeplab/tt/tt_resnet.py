# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Union
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_stem import TtStem
from models.experimental.panoptic_deeplab.tt.tt_bottleneck import TtBottleneck
from models.common.lightweightmodule import LightweightModule


class TtResNet(LightweightModule):
    """
    TTNN implementation of ResNet backbone for Panoptic DeepLab using TT CNN Builder API.

    Architecture:
    - stem: DeepLabStem (3 conv layers + maxpool)
    - res2: 3 BottleneckBlocks, stride=1
    - res3: 4 BottleneckBlocks, first has stride=2
    - res4: 6 BottleneckBlocks, first has stride=2
    - res5: 3 BottleneckBlocks, dilated convolutions (dilation=2,4,8)

    This implementation uses the TT CNN Builder API for all convolutional and pooling
    layers. Layer configurations (Conv2dConfiguration, MaxPool2dConfiguration) are
    extracted from the PyTorch model during preprocessing and can be customized via
    the model_configs parameter.

    Args:
        parameters: Preprocessed model parameters containing Conv2dConfiguration and
                   MaxPool2dConfiguration objects for each layer
        device: TTNN device
        dtype: Either a single DataType to apply to all layers, or a dict mapping
               layer names ("stem", "res2", "res3", "res4", "res5") to DataTypes
               for per-layer precision control
        model_configs: ModelOptimisations instance for applying layer-specific config
                      overrides (slicing strategies, sharding strategies, etc.)
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: Union[ttnn.DataType, dict[str, ttnn.DataType]] = ttnn.bfloat8_b,
        model_configs=None,
    ):
        super().__init__()
        self.device = device
        self.model_configs = model_configs

        logger.debug("Initializing TtResNet")

        # Handle dtype parameter - if it's a dict, use it as layer_dtypes, otherwise apply to all layers
        if isinstance(dtype, dict):
            layer_dtypes = dtype
        else:
            layer_dtypes = {
                "stem": dtype,
                "res2": dtype,
                "res3": dtype,
                "res4": dtype,
                "res5": dtype,
            }

        # Initialize stem
        stem_dtype = layer_dtypes.get("stem", dtype)
        self.stem = TtStem(parameters=parameters["stem"], device=device, dtype=stem_dtype, model_configs=model_configs)
        logger.debug(f"Stem initialized with dtype: {stem_dtype}")

        # Initialize residual layers with per-layer dtypes
        res2_dtype = layer_dtypes.get("res2", dtype)
        self.res2 = self._build_res_layer("res2", parameters["res2"], device, res2_dtype, 3, stride=1)
        logger.debug(f"Res2 initialized with dtype: {res2_dtype}")

        res3_dtype = layer_dtypes.get("res3", dtype)
        self.res3 = self._build_res_layer("res3", parameters["res3"], device, res3_dtype, 4, stride=2)
        logger.debug(f"Res3 initialized with dtype: {res3_dtype}")

        res4_dtype = layer_dtypes.get("res4", dtype)
        self.res4 = self._build_res_layer("res4", parameters["res4"], device, res4_dtype, 6, stride=2)
        logger.debug(f"Res4 initialized with dtype: {res4_dtype}")

        res5_dtype = layer_dtypes.get("res5", dtype)
        self.res5 = self._build_res_layer("res5", parameters["res5"], device, res5_dtype, 3, dilations=[2, 4, 8])
        logger.debug(f"Res5 initialized with dtype: {res5_dtype}")

        logger.debug("TtResNet initialization complete")

    def _build_res_layer(self, layer_name, layer_params, device, dtype, num_blocks, stride=1, dilations=None):
        """Build a residual layer with the specified configuration"""
        blocks = []

        for i in range(num_blocks):
            block_params = layer_params[i]
            has_shortcut = i == 0
            block_stride = stride if i == 0 else 1
            shortcut_stride = stride if i == 0 else 1
            dilation = dilations[i] if dilations else 1
            block_id = f"{layer_name}.{i}"

            blocks.append(
                TtBottleneck(
                    parameters=block_params,
                    device=device,
                    dtype=dtype,
                    has_shortcut=has_shortcut,
                    stride=block_stride,
                    dilation=dilation,
                    shortcut_stride=shortcut_stride,
                    block_id=block_id,
                    model_configs=self.model_configs,
                )
            )

        logger.debug(f"{layer_name} layer initialized ({num_blocks} blocks)")
        return blocks

    def forward(self, x: ttnn.Tensor) -> dict[str, ttnn.Tensor]:
        """
        Forward pass through ResNet backbone.

        Returns dictionary with feature maps from res2, res3, res4, res5.
        """
        logger.debug(f"TtResNet forward - input: {x.shape}")

        # Stem processing
        x = self.stem(x)
        logger.debug(f"Stem complete - output: {x.shape}")

        # Spatial dimensions for each stage (derived from model architecture)
        # res2: H/4 x W/4, res3: H/8 x W/8, res4: H/16 x W/16, res5: H/16 x W/16
        spatial_dims = {
            "res2": (128, 256),  # For 512x1024 input
            "res3": (64, 128),
            "res4": (32, 64),
            "res5": (32, 64),
        }

        # Process residual layers and collect outputs
        outputs = {}
        for layer_name, layer in [("res2", self.res2), ("res3", self.res3), ("res4", self.res4), ("res5", self.res5)]:
            x = self._forward_res_layer(x, layer)
            if x.is_sharded():
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            else:
                x = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Reshape if flattened [1, 1, H*W, C] -> [1, H, W, C]
            expected_h, expected_w = spatial_dims[layer_name]
            if x.shape[1] == 1 and x.shape[2] == expected_h * expected_w:
                logger.debug(f"{layer_name}: Reshaping from {x.shape} to [1, {expected_h}, {expected_w}, {x.shape[3]}]")
                x = ttnn.reshape(x, (1, expected_h, expected_w, x.shape[3]))

            # Clone the output to store independently (backbone outputs are shared between heads)
            # This prevents deallocation in subsequent stages from affecting stored outputs
            outputs[layer_name] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logger.debug(f"{layer_name} complete - output: {outputs[layer_name].shape}")

        return outputs

    def _forward_res_layer(self, x: ttnn.Tensor, layer: list[TtBottleneck]) -> ttnn.Tensor:
        """Forward pass through a residual layer"""
        for block in layer:
            x = block(x)
        return x

    def forward_single_output(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass returning only the final output (res5)."""
        x = self.stem(x)

        for layer in [self.res2, self.res3, self.res4, self.res5]:
            x = self._forward_res_layer(x, layer)

        return x
