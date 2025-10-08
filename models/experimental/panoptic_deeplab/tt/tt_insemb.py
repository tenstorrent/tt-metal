# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import TtConv2d, TtConv2dParameters
from models.experimental.panoptic_deeplab.tt.tt_upsample_wrapper import TtUpsample
from models.experimental.panoptic_deeplab.tt.tt_semseg import TtDeepLabV3PlusHead
from models.experimental.panoptic_deeplab.reference.pytorch_semseg import ShapeSpec


class TtPanopticDeepLabInsEmbedHead(TtDeepLabV3PlusHead):
    """
    TTNN implementation for Panoptic-DeepLab instance embedding head.
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

        # --- Center Prediction Branch ---
        # center_head_0
        # Handle both dict and object parameter formats
        center_head_params = parameters["center_head"] if isinstance(parameters, dict) else parameters.center_head
        center_head0_path = center_head_params[0]
        ch0_bias = (
            center_head0_path["bias"]
            if isinstance(center_head0_path, dict) and "bias" in center_head0_path
            else (center_head0_path.bias if hasattr(center_head0_path, "bias") else None)
        )
        ch0_weight = center_head0_path["weight"] if isinstance(center_head0_path, dict) else center_head0_path.weight
        ch0_params = TtConv2dParameters(
            weight=ch0_weight,
            bias=ch0_bias,
            device=self.device,
        )
        # Get slice config from model_configs for center_head.0
        if self.model_configs is not None:
            head_slice_config = self.model_configs.get_head_slice_config("center")
            self.center_head_0 = TtConv2d.create_with_height_slicing(
                ch0_params,
                num_slices=head_slice_config["num_slices"],
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.center_head.0",
                model_configs=self.model_configs,
            )
        else:
            # Fallback to original logic
            logger.warning(
                "FALLBACK INSEMB CENTER_HEAD.0: Using original hardcoded logic for instance_head.center_head.0 instead of model_configs"
            )
            self.center_head_0 = TtConv2d.create_with_height_slicing(
                ch0_params,
                num_slices=2,
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.center_head.0",
                model_configs=self.model_configs,
            )

        # center_head_1
        center_head1_path = center_head_params[1]
        ch1_bias = (
            center_head1_path["bias"]
            if isinstance(center_head1_path, dict) and "bias" in center_head1_path
            else (center_head1_path.bias if hasattr(center_head1_path, "bias") else None)
        )
        ch1_weight = center_head1_path["weight"] if isinstance(center_head1_path, dict) else center_head1_path.weight
        ch1_params = TtConv2dParameters(
            weight=ch1_weight,
            bias=ch1_bias,
            device=self.device,
        )
        # Get slice config from model_configs for center_head.1
        if self.model_configs is not None:
            head_slice_config = self.model_configs.get_head_slice_config("center")
            self.center_head_1 = TtConv2d.create_with_height_slicing(
                ch1_params,
                num_slices=head_slice_config["num_slices"],
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.center_head.1",
                model_configs=self.model_configs,
            )
        else:
            # Fallback to original logic
            logger.warning(
                "FALLBACK INSEMB CENTER_HEAD.1: Using original hardcoded logic for instance_head.center_head.1 instead of model_configs"
            )
            self.center_head_1 = TtConv2d.create_with_height_slicing(
                ch1_params,
                num_slices=2,
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.center_head.1",
                model_configs=self.model_configs,
            )

        # center_predictor
        center_predictor_params = (
            parameters["center_predictor"] if isinstance(parameters, dict) else parameters.center_predictor
        )
        center_predictor_path = center_predictor_params
        cp_bias = (
            center_predictor_path["bias"]
            if isinstance(center_predictor_path, dict) and "bias" in center_predictor_path
            else (center_predictor_path.bias if hasattr(center_predictor_path, "bias") else None)
        )
        cp_weight = (
            center_predictor_path["weight"] if isinstance(center_predictor_path, dict) else center_predictor_path.weight
        )
        cp_params = TtConv2dParameters(
            weight=cp_weight,
            bias=cp_bias,
            device=self.device,
        )
        self.center_predictor = TtConv2d.create(
            cp_params,
            stride=(1, 1),
            padding=(0, 0),
            conv_path="instance_head.center_predictor",
            model_configs=self.model_configs,
        )

        # --- Offset Prediction Branch ---
        # offset_head_0
        offset_head_params = parameters["offset_head"] if isinstance(parameters, dict) else parameters.offset_head
        offset_head0_path = offset_head_params[0]
        oh0_bias = (
            offset_head0_path["bias"]
            if isinstance(offset_head0_path, dict) and "bias" in offset_head0_path
            else (offset_head0_path.bias if hasattr(offset_head0_path, "bias") else None)
        )
        oh0_weight = offset_head0_path["weight"] if isinstance(offset_head0_path, dict) else offset_head0_path.weight
        oh0_params = TtConv2dParameters(
            weight=oh0_weight,
            bias=oh0_bias,
            device=self.device,
        )
        # Get slice config from model_configs for offset_head.0
        if self.model_configs is not None:
            head_slice_config = self.model_configs.get_head_slice_config("offset")
            self.offset_head_0 = TtConv2d.create_with_height_slicing(
                oh0_params,
                num_slices=head_slice_config["num_slices"],
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.offset_head.0",
                model_configs=self.model_configs,
            )
        else:
            # Fallback to original logic
            logger.warning(
                "FALLBACK INSEMB OFFSET_HEAD.0: Using original hardcoded logic for instance_head.offset_head.0 instead of model_configs"
            )
            self.offset_head_0 = TtConv2d.create_with_height_slicing(
                oh0_params,
                num_slices=2,
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.offset_head.0",
                model_configs=self.model_configs,
            )

        # offset_head_1
        offset_head1_path = offset_head_params[1]
        oh1_bias = (
            offset_head1_path["bias"]
            if isinstance(offset_head1_path, dict) and "bias" in offset_head1_path
            else (offset_head1_path.bias if hasattr(offset_head1_path, "bias") else None)
        )
        oh1_weight = offset_head1_path["weight"] if isinstance(offset_head1_path, dict) else offset_head1_path.weight
        oh1_params = TtConv2dParameters(
            weight=oh1_weight,
            bias=oh1_bias,
            device=self.device,
        )
        # Get slice config from model_configs for offset_head.1
        if self.model_configs is not None:
            head_slice_config = self.model_configs.get_head_slice_config("offset")
            self.offset_head_1 = TtConv2d.create_with_height_slicing(
                oh1_params,
                num_slices=head_slice_config["num_slices"],
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.offset_head.1",
                model_configs=self.model_configs,
            )
        else:
            # Fallback to original logic
            logger.warning(
                "FALLBACK INSEMB OFFSET_HEAD.1: Using original hardcoded logic for instance_head.offset_head.1 instead of model_configs"
            )
            self.offset_head_1 = TtConv2d.create_with_height_slicing(
                oh1_params,
                num_slices=2,
                stride=(1, 1),
                padding=(1, 1),
                conv_path="instance_head.offset_head.1",
                model_configs=self.model_configs,
            )

        # offset_predictor
        offset_predictor_params = (
            parameters["offset_predictor"] if isinstance(parameters, dict) else parameters.offset_predictor
        )
        offset_predictor_path = offset_predictor_params
        op_bias = (
            offset_predictor_path["bias"]
            if isinstance(offset_predictor_path, dict) and "bias" in offset_predictor_path
            else (offset_predictor_path.bias if hasattr(offset_predictor_path, "bias") else None)
        )
        op_weight = (
            offset_predictor_path["weight"] if isinstance(offset_predictor_path, dict) else offset_predictor_path.weight
        )
        op_params = TtConv2dParameters(
            weight=op_weight,
            bias=op_bias,
            device=self.device,
        )
        self.offset_predictor = TtConv2d.create(
            op_params,
            stride=(1, 1),
            padding=(0, 0),
            conv_path="instance_head.offset_predictor",
            model_configs=self.model_configs,
        )

        self.final_upsample = TtUpsample.create(device=device, scale_factor=common_stride, mode="nearest")
        logger.debug("TtPanopticDeepLabInsEmbedHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor, Dict, Dict]:
        logger.debug("TtPanopticDeepLabInsEmbedHead forward pass starting")
        center_logits, offset_logits = self.layers(features)

        # --- Final Upsample for Center ---
        center_logits = self.final_upsample(center_logits)
        logger.debug(f"TtPanopticDeepLabInsEmbedHead center upsample complete - shape: {center_logits.shape}")

        # --- Final Upsample for Offset ---
        offset_logits = self.final_upsample(offset_logits)
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
