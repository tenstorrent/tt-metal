# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple
import ttnn
from loguru import logger

from models.tt_cnn.tt.builder import TtUpsample, UpsampleConfiguration
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
        norm: str,
        train_size: Optional[Tuple],
        model_configs=None,
    ):
        # Handle both dict and object parameter formats
        decoder_params = parameters["decoder"] if isinstance(parameters, dict) else parameters.decoder
        super().__init__(
            parameters=decoder_params,
            device=device,
            input_shape=input_shape,
            norm=norm,
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
        use_bias = norm == ""
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

        # --- Offset Prediction Branch - use builder API ---
        offset_head_params = parameters["offset_head"] if isinstance(parameters, dict) else parameters.offset_head
        self.offset_head_0 = self._create_conv_layer(offset_head_params[0], "instance_head.offset_head.0")
        self.offset_head_1 = self._create_conv_layer(offset_head_params[1], "instance_head.offset_head.1")

        offset_predictor_params = (
            parameters["offset_predictor"] if isinstance(parameters, dict) else parameters.offset_predictor
        )
        self.offset_predictor = self._create_conv_layer(offset_predictor_params, "instance_head.offset_predictor")

        # Final upsample - use builder API
        # Store scale factor for dynamic upsample creation during forward pass
        self.final_upsample_scale = (
            common_stride if isinstance(common_stride, tuple) else (common_stride, common_stride)
        )
        self.final_upsample_mode = "nearest"
        logger.debug("TtPanopticDeepLabInsEmbedHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor, Dict, Dict]:
        logger.debug("TtPanopticDeepLabInsEmbedHead forward pass starting")
        center_logits, offset_logits = self.layers(features)

        # --- Final Upsample for Center ---
        center_upsample_config = UpsampleConfiguration(
            scale_factor=self.final_upsample_scale,
            mode=self.final_upsample_mode,
            input_height=center_logits.shape[1],
            input_width=center_logits.shape[2],
            batch_size=center_logits.shape[0],
            channels=center_logits.shape[3],
        )
        center_upsample = TtUpsample(center_upsample_config, self.device)
        center_logits = center_upsample(center_logits)
        logger.debug(f"TtPanopticDeepLabInsEmbedHead center upsample complete - shape: {center_logits.shape}")

        # --- Final Upsample for Offset ---
        offset_upsample_config = UpsampleConfiguration(
            scale_factor=self.final_upsample_scale,
            mode=self.final_upsample_mode,
            input_height=offset_logits.shape[1],
            input_width=offset_logits.shape[2],
            batch_size=offset_logits.shape[0],
            channels=offset_logits.shape[3],
        )
        offset_upsample = TtUpsample(offset_upsample_config, self.device)
        offset_logits = offset_upsample(offset_logits)
        offset_logits = ttnn.mul(offset_logits, self.common_stride)
        logger.debug(f"TtPanopticDeepLabInsEmbedHead offset upsample complete - shape: {offset_logits.shape}")

        logger.debug("TtPanopticDeepLabInsEmbedHead forward pass complete")
        return center_logits, offset_logits, {}, {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        y = super().layers(features)

        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

        # --- Center Prediction Branch ---
        center_y = self.center_head_0(y)
        center_y = self.activation(center_y)
        center_y = self.center_head_1(center_y)
        center_y = self.activation(center_y)
        center_logits = self.center_predictor(center_y)

        # --- Offset Prediction Branch ---
        offset_y = self.offset_head_0(y)
        offset_y = self.activation(offset_y)
        offset_y = self.offset_head_1(offset_y)
        offset_y = self.activation(offset_y)
        offset_logits = self.offset_predictor(offset_y)

        return center_logits, offset_logits
