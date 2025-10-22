# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_semseg import TtDeepLabV3PlusHead
from models.experimental.panoptic_deeplab.reference.pytorch_semseg import ShapeSpec


class TtPanopticDeepLabInsEmbedHead(TtDeepLabV3PlusHead):
    """
    TTNN implementation for Panoptic-DeepLab instance embedding head.
    Uses TT CNN Builder API for convolutions and upsampling.

    Inherits from TtDeepLabV3PlusHead which provides the decoder and _create_conv_layer helper.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.Device,
        *,
        input_shape: Dict[str, ShapeSpec],
        head_channels: int,
        project_channels,
        aspp_dilations,
        aspp_dropout: float,
        decoder_channels,
        common_stride: int,
        train_size: Optional[Tuple],
        model_configs=None,
    ):
        # Handle both dict and object parameter formats
        decoder_params = parameters["decoder"] if isinstance(parameters, dict) else parameters.decoder
        super().__init__(
            parameters=decoder_params,
            device=device,
            input_shape=input_shape,
            num_classes=None,  # decoder_only mode
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            train_size=train_size,
            model_configs=model_configs,
        )
        assert self.decoder_only
        decoder_out_ch = decoder_channels[0]
        logger.debug(f"Initializing TtPanopticDeepLabInsEmbedHead with head_channels: {head_channels}")

        # --- Center Prediction Branch - use builder API ---
        center_head_params = parameters["center_head"] if isinstance(parameters, dict) else parameters.center_head
        self.center_head_0 = self._create_conv_layer(center_head_params[0], "instance_head.center_head.0")
        self.center_head_1 = self._create_conv_layer(center_head_params[1], "instance_head.center_head.1")

        center_predictor_params = (
            parameters["center_predictor"] if isinstance(parameters, dict) else parameters.center_predictor
        )
        self.center_predictor = self._create_conv_layer(center_predictor_params, "instance_head.center_predictor")

        # Store original output channels if padding was applied during preprocessing
        if isinstance(center_predictor_params, dict) and "original_out_channels" in center_predictor_params:
            self._center_output_original_channels = center_predictor_params["original_out_channels"]
            logger.debug(f"Center predictor: stored original_out_channels={self._center_output_original_channels}")

        # --- Offset Prediction Branch - use builder API ---
        offset_head_params = parameters["offset_head"] if isinstance(parameters, dict) else parameters.offset_head
        self.offset_head_0 = self._create_conv_layer(offset_head_params[0], "instance_head.offset_head.0")
        self.offset_head_1 = self._create_conv_layer(offset_head_params[1], "instance_head.offset_head.1")

        offset_predictor_params = (
            parameters["offset_predictor"] if isinstance(parameters, dict) else parameters.offset_predictor
        )
        self.offset_predictor = self._create_conv_layer(offset_predictor_params, "instance_head.offset_predictor")

        # Store original output channels if padding was applied during preprocessing
        if isinstance(offset_predictor_params, dict) and "original_out_channels" in offset_predictor_params:
            self._offset_output_original_channels = offset_predictor_params["original_out_channels"]
            logger.debug(f"Offset predictor: stored original_out_channels={self._offset_output_original_channels}")

        # Final upsample - use builder API
        # Store scale factor for dynamic upsample creation during forward pass
        self.final_upsample_scale = (
            common_stride if isinstance(common_stride, tuple) else (common_stride, common_stride)
        )
        self.final_upsample_mode = "nearest"

        # Initialize original output channels to None if not already set from parameters
        if not hasattr(self, "_center_output_original_channels"):
            self._center_output_original_channels = None
        if not hasattr(self, "_offset_output_original_channels"):
            self._offset_output_original_channels = None

        # Store spatial dimensions for upsample (will be set during forward pass)
        self._center_predictor_h = None
        self._center_predictor_w = None
        self._offset_predictor_h = None
        self._offset_predictor_w = None

        logger.debug("TtPanopticDeepLabInsEmbedHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor, Dict, Dict]:
        logger.debug("TtPanopticDeepLabInsEmbedHead forward pass starting")
        center_logits, offset_logits = self.layers(features)

        # --- Final Upsample for Center ---
        # Use saved spatial dimensions (predictors may output in TILE_LAYOUT with padding)
        current_h = self._center_predictor_h
        current_w = self._center_predictor_w

        # If flattened, reshape before upsample
        if center_logits.shape[1] == 1 and center_logits.shape[2] == current_h * current_w:
            logger.debug(
                f"Reshaping flattened center output from {center_logits.shape} to [1, {current_h}, {current_w}, {center_logits.shape[3]}]"
            )
            center_logits = ttnn.reshape(
                center_logits, (center_logits.shape[0], current_h, current_w, center_logits.shape[3])
            )

        # Convert to interleaved DRAM if sharded
        if center_logits.is_sharded():
            center_logits = ttnn.sharded_to_interleaved(center_logits, ttnn.DRAM_MEMORY_CONFIG)
        else:
            center_logits = ttnn.to_memory_config(center_logits, ttnn.DRAM_MEMORY_CONFIG)

        # Convert to ROW_MAJOR for upsample
        center_logits = ttnn.to_layout(center_logits, ttnn.ROW_MAJOR_LAYOUT)

        # Calculate scale factors
        scale_h = self.final_upsample_scale[0]
        scale_w = self.final_upsample_scale[1]
        logger.debug(
            f"Center: Upsampling from [{current_h}, {current_w}] with scale_factor=[{scale_h}, {scale_w}] to [{current_h * scale_h}, {current_w * scale_w}]"
        )

        # Upsample directly
        center_logits = ttnn.upsample(center_logits, scale_factor=(scale_h, scale_w), mode=self.final_upsample_mode)

        # Convert back to TILE_LAYOUT and DRAM
        center_logits = ttnn.to_layout(center_logits, ttnn.TILE_LAYOUT)
        center_logits = ttnn.to_memory_config(center_logits, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtPanopticDeepLabInsEmbedHead center upsample complete - shape: {center_logits.shape}")

        # --- Final Upsample for Offset ---
        # Use saved spatial dimensions
        current_h = self._offset_predictor_h
        current_w = self._offset_predictor_w

        # If flattened, reshape before upsample
        if offset_logits.shape[1] == 1 and offset_logits.shape[2] == current_h * current_w:
            logger.debug(
                f"Reshaping flattened offset output from {offset_logits.shape} to [1, {current_h}, {current_w}, {offset_logits.shape[3]}]"
            )
            offset_logits = ttnn.reshape(
                offset_logits, (offset_logits.shape[0], current_h, current_w, offset_logits.shape[3])
            )

        # Convert to interleaved DRAM if sharded
        if offset_logits.is_sharded():
            offset_logits = ttnn.sharded_to_interleaved(offset_logits, ttnn.DRAM_MEMORY_CONFIG)
        else:
            offset_logits = ttnn.to_memory_config(offset_logits, ttnn.DRAM_MEMORY_CONFIG)

        # Convert to ROW_MAJOR for upsample
        offset_logits = ttnn.to_layout(offset_logits, ttnn.ROW_MAJOR_LAYOUT)

        # Calculate scale factors
        scale_h = self.final_upsample_scale[0]
        scale_w = self.final_upsample_scale[1]
        logger.debug(
            f"Offset: Upsampling from [{current_h}, {current_w}] with scale_factor=[{scale_h}, {scale_w}] to [{current_h * scale_h}, {current_w * scale_w}]"
        )

        # Upsample directly
        offset_logits = ttnn.upsample(offset_logits, scale_factor=(scale_h, scale_w), mode=self.final_upsample_mode)

        # Convert back to TILE_LAYOUT and DRAM
        offset_logits = ttnn.to_layout(offset_logits, ttnn.TILE_LAYOUT)
        offset_logits = ttnn.to_memory_config(offset_logits, ttnn.DRAM_MEMORY_CONFIG)

        # Apply offset scaling
        offset_logits = ttnn.mul(offset_logits, self.common_stride)
        logger.debug(f"TtPanopticDeepLabInsEmbedHead offset upsample complete - shape: {offset_logits.shape}")

        logger.debug("TtPanopticDeepLabInsEmbedHead forward pass complete")
        return center_logits, offset_logits, {}, {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        y = super().layers(features)

        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

        # Save spatial dimensions for upsample (head convs have stride=1)
        self._center_predictor_h = y.shape[1]
        self._center_predictor_w = y.shape[2]
        self._offset_predictor_h = y.shape[1]
        self._offset_predictor_w = y.shape[2]
        logger.debug(f"Saved predictor spatial dimensions: H={self._center_predictor_h}, W={self._center_predictor_w}")

        # --- Center Prediction Branch ---
        logger.info(f"🔷 Executing conv: instance_head.center_head.0")
        center_y = self.center_head_0(y)

        logger.info(f"🔷 Executing conv: instance_head.center_head.1")
        center_y = self.center_head_1(center_y)

        # Convert to interleaved for predictor
        if center_y.is_sharded():
            center_y = ttnn.sharded_to_interleaved(center_y, ttnn.DRAM_MEMORY_CONFIG)
        else:
            center_y = ttnn.to_memory_config(center_y, ttnn.DRAM_MEMORY_CONFIG)

        logger.info(f"🔷 Executing conv: instance_head.center_predictor")
        center_logits = self.center_predictor(center_y)

        # --- Offset Prediction Branch ---
        logger.info(f"🔷 Executing conv: instance_head.offset_head.0")
        offset_y = self.offset_head_0(y)

        logger.info(f"🔷 Executing conv: instance_head.offset_head.1")
        offset_y = self.offset_head_1(offset_y)

        # Convert to interleaved for predictor
        if offset_y.is_sharded():
            offset_y = ttnn.sharded_to_interleaved(offset_y, ttnn.DRAM_MEMORY_CONFIG)
        else:
            offset_y = ttnn.to_memory_config(offset_y, ttnn.DRAM_MEMORY_CONFIG)

        logger.info(f"🔷 Executing conv: instance_head.offset_predictor")
        offset_logits = self.offset_predictor(offset_y)

        return center_logits, offset_logits

    def get_center_output_channels_for_slicing(self):
        """Return the original (unpadded) center output channel count, or None if no padding was applied."""
        return self._center_output_original_channels

    def get_offset_output_channels_for_slicing(self):
        """Return the original (unpadded) offset output channel count, or None if no padding was applied."""
        return self._offset_output_original_channels
