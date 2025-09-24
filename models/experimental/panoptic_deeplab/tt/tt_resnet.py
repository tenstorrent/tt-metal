# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_stem import TtStem
from models.experimental.panoptic_deeplab.tt.tt_bottleneck import TtBottleneck
from models.common.lightweightmodule import LightweightModule


class TtResNet(LightweightModule):
    """
    TTNN implementation of ResNet backbone for Panoptic DeepLab.

    Architecture:
    - stem: DeepLabStem (3 conv layers)
    - res2: 3 blocks, stride=1
    - res3: 4 blocks, first has stride=2
    - res4: 6 blocks, first has stride=2
    - res5: 3 blocks, dilated convolutions (2, 4, 8)
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device

        logger.debug("Initializing TtResNet")

        # Initialize stem
        self.stem = TtStem(parameters=parameters["stem"], device=device, dtype=dtype)

        # Initialize residual layers
        self.res2 = self._build_res_layer("res2", parameters["res2"], device, dtype, 3, stride=1)
        self.res3 = self._build_res_layer("res3", parameters["res3"], device, dtype, 4, stride=2)
        self.res4 = self._build_res_layer("res4", parameters["res4"], device, dtype, 6, stride=2)
        self.res5 = self._build_res_layer("res5", parameters["res5"], device, dtype, 3, dilations=[2, 4, 8])

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

        # Process residual layers and collect outputs
        outputs = {}
        for layer_name, layer in [("res2", self.res2), ("res3", self.res3), ("res4", self.res4), ("res5", self.res5)]:
            x = self._forward_res_layer(x, layer)
            outputs[layer_name] = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            logger.debug(f"{layer_name} complete - output: {outputs[layer_name].shape}")

        return outputs

    def _forward_res_layer(self, x: ttnn.Tensor, layer: []) -> ttnn.Tensor:
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
