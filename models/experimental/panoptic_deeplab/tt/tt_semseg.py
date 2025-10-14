# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Union, Optional, Tuple
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_aspp import TtASPP, get_ttnn_activation
from models.tt_cnn.tt.builder import TtConv2d, TtUpsample
from models.experimental.panoptic_deeplab.reference.pytorch_semseg import ShapeSpec
from models.common.lightweightmodule import LightweightModule


class TtDeepLabV3PlusHead(LightweightModule):
    """
    TTNN implementation of the DeepLabV3+ segmentation head.
    Uses TT CNN Builder API for convolutions and upsampling.

    Conv2dConfiguration objects are extracted during preprocessing and stored
    in parameters dict. This class applies model-specific overrides via model_configs.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.Device,
        *,
        input_shape: Dict[str, ShapeSpec],
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: str,
        train_size: Optional[Tuple],
        num_classes: Optional[int] = None,
        model_configs=None,
    ):
        super().__init__()
        sorted_input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted_input_shape]
        in_channels = [v.channels for k, v in sorted_input_shape]
        in_strides = [v.stride for k, v in sorted_input_shape]
        aspp_channels = decoder_channels[-1]

        logger.debug(
            f"Initializing TtDeepLabV3PlusHead with features: {self.in_features}, decoder_only: {num_classes is None}"
        )

        self.common_stride = common_stride
        self.decoder_only = num_classes is None
        self.device = device
        self.model_configs = model_configs
        self.activation = get_ttnn_activation("relu")

        # Initialize decoder with all stages
        self._initialize_decoder(
            parameters,
            in_channels,
            in_strides,
            project_channels,
            decoder_channels,
            aspp_dilations,
            aspp_dropout,
            norm,
            train_size,
        )

    def _create_conv_layer(self, params, conv_path: str):
        """Helper to create conv layer using TT CNN Builder with config overrides"""
        if "conv_config" in params:
            base_config = params["conv_config"]
            logger.debug(f"Using Conv2dConfiguration from preprocessing for {conv_path}")
        else:
            logger.error(f"Conv2dConfiguration not found for {conv_path}")
            raise ValueError(
                f"Expected 'conv_config' in parameters for {conv_path}. Please use new preprocessing system."
            )

        # Apply model-specific overrides if model_configs is provided
        if self.model_configs is not None:
            final_config = self.model_configs.apply_conv_overrides(base_config, conv_path=conv_path)
            logger.debug(f"Applied config overrides for {conv_path}")
        else:
            final_config = base_config
            logger.debug(f"No model_configs provided, using base config for {conv_path}")

        # Create TtConv2d using TT CNN Builder
        return TtConv2d(final_config, self.device)

    def _initialize_decoder(
        self,
        parameters,
        in_channels,
        in_strides,
        project_channels,
        decoder_channels,
        aspp_dilations,
        aspp_dropout,
        norm,
        train_size,
    ):
        """Initialize decoder stages"""
        self.decoder = {}
        use_bias = norm == ""

        for idx, in_channel in enumerate(in_channels):
            decoder_stage = {}
            feature_name = self.in_features[idx]

            if idx == len(self.in_features) - 1:
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    pool_h, pool_w = train_h // encoder_stride, train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None

                # Handle both dict and object parameter formats
                feature_params = (
                    parameters[feature_name] if isinstance(parameters, dict) else getattr(parameters, feature_name)
                )
                project_conv_params = (
                    feature_params["project_conv"] if isinstance(feature_params, dict) else feature_params.project_conv
                )
                aspp_channels = decoder_channels[-1]
                project_conv = TtASPP(
                    in_channels=in_channel,
                    out_channels=aspp_channels,
                    dilations=aspp_dilations,
                    device=self.device,
                    pool_kernel_size=pool_kernel_size,
                    norm=norm,
                    activation="relu",
                    dropout=aspp_dropout,
                    parameters=project_conv_params,
                    model_configs=self.model_configs,
                )
                decoder_stage["project_conv"] = project_conv
                decoder_stage["fuse_conv_0"] = None
            else:
                proj_out_ch = project_channels[idx]
                fuse_out_ch = decoder_channels[idx]

                # Parameter path for this decoder stage
                base_path = parameters[feature_name]

                # Project Conv - use builder API
                proj_conv_params = base_path["project_conv"] if isinstance(base_path, dict) else base_path.project_conv
                project_conv = self._create_conv_layer(proj_conv_params, f"decoder.{feature_name}.project_conv")

                # Fuse Conv 0 - use builder API
                fuse_conv_params = base_path["fuse_conv"] if isinstance(base_path, dict) else base_path.fuse_conv
                fuse_conv_0 = self._create_conv_layer(fuse_conv_params[0], f"decoder.{feature_name}.fuse_conv.0")

                # Fuse Conv 1 - use builder API
                fuse_conv_1 = self._create_conv_layer(fuse_conv_params[1], f"decoder.{feature_name}.fuse_conv.1")

                decoder_stage["project_conv"] = project_conv
                decoder_stage["fuse_conv_0"] = fuse_conv_0
                decoder_stage["fuse_conv_1"] = fuse_conv_1

            self.decoder[feature_name] = decoder_stage

        # Initialize upsample operations for decoder using builder API
        # Upsamples will be created dynamically during forward pass with the correct shapes
        logger.debug("TtDeepLabV3PlusHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Union[ttnn.Tensor, Tuple[ttnn.Tensor, Dict]]:
        y = self.layers(features)
        return y

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """
        Executes the decoder pipeline, mirroring the PyTorch version's logic.
        Simplified to use single conv versions - slicing is handled by Conv2dConfiguration.
        """
        logger.debug(f"TtDeepLabV3PlusHead layers - processing features: {list(features.keys())}")
        y = None
        feature_keys = self.in_features[::-1]

        # --- Stage 1: ASPP ---
        aspp_feature_key = feature_keys[0]
        x = features[aspp_feature_key]
        stage = self.decoder[aspp_feature_key]
        logger.debug(
            f"TtDeepLabV3PlusHead processing ASPP stage with feature: {aspp_feature_key}, input shape: {x.shape}"
        )
        y = stage["project_conv"](x)
        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtDeepLabV3PlusHead ASPP stage complete, output shape: {y.shape}")

        # --- Subsequent Fusion Stages (e.g., 'res3', then 'res2') ---
        for i, f_key in enumerate(feature_keys[1:]):
            logger.debug(f"TtDeepLabV3PlusHead processing fusion stage {i+1} with feature: {f_key}")
            previous_y = y
            x = features[f_key]
            stage = self.decoder[f_key]
            proj_x = stage["project_conv"](x)
            proj_x = self.activation(proj_x)
            proj_x = ttnn.to_memory_config(proj_x, ttnn.DRAM_MEMORY_CONFIG)
            logger.debug(f"TtDeepLabV3PlusHead fusion stage {i+1} projection complete, shape: {proj_x.shape}")

            scale_h = proj_x.shape[1] // y.shape[1]
            scale_w = proj_x.shape[2] // y.shape[2]

            # Update upsample scale and perform upsampling
            # Note: TtUpsample with builder API might need dynamic scale update
            # For now, we'll use the same approach but with a single upsample instance
            from models.tt_cnn.tt.builder import UpsampleConfiguration

            upsample_config = UpsampleConfiguration(
                scale_factor=(scale_h, scale_w),
                mode="bilinear",
                input_height=y.shape[1],
                input_width=y.shape[2],
                batch_size=y.shape[0],
                in_channels=y.shape[3],
            )
            upsample_op = TtUpsample(upsample_config, self.device)
            y_upsampled = upsample_op(y)

            y_upsampled = ttnn.to_memory_config(y_upsampled, ttnn.DRAM_MEMORY_CONFIG)
            y_upsampled = ttnn.to_layout(y_upsampled, ttnn.TILE_LAYOUT)

            # Ensure both tensors have the same dtype before concatenation
            target_dtype = ttnn.bfloat8_b
            if proj_x.dtype != target_dtype:
                proj_x = ttnn.typecast(proj_x, target_dtype)
            if y_upsampled.dtype != target_dtype:
                y_upsampled = ttnn.typecast(y_upsampled, target_dtype)

            y = ttnn.concat([proj_x, y_upsampled], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            ttnn.deallocate(previous_y)
            ttnn.deallocate(proj_x)
            ttnn.deallocate(y_upsampled)

            # Use single conv versions - slicing handled by config
            y_conv0 = stage["fuse_conv_0"](y)
            ttnn.deallocate(y)
            y_act0 = self.activation(y_conv0)
            ttnn.deallocate(y_conv0)

            ttnn.to_memory_config(y_act0, ttnn.DRAM_MEMORY_CONFIG)
            y_conv1 = stage["fuse_conv_1"](y_act0)
            y = self.activation(y_conv1)
            ttnn.deallocate(y_conv1)

        logger.debug(f"TtDeepLabV3PlusHead layers complete - final output shape: {y.shape}")
        return y


class TtPanopticDeepLabSemSegHead(TtDeepLabV3PlusHead):
    """
    TTNN implementation of the Panoptic-DeepLab semantic segmentation head.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.Device,
        *,
        input_shape: Dict[str, ShapeSpec],
        head_channels: int,
        num_classes: int,
        norm: str,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
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
            norm=norm,
            num_classes=None,
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
        logger.debug(f"Initializing TtPanopticDeepLabSemSegHead with {num_classes} classes")

        # Head 0 - use builder API
        head_params = parameters["head"] if isinstance(parameters, dict) else parameters.head
        self.head_0 = self._create_conv_layer(head_params[0], "semantic_head.head.0")

        # Head 1 - use builder API
        self.head_1 = self._create_conv_layer(head_params[1], "semantic_head.head.1")

        # Predictor - use builder API
        predictor_params = parameters["predictor"] if isinstance(parameters, dict) else parameters.predictor
        self.predictor = self._create_conv_layer(predictor_params, "semantic_head.predictor")

        # Final upsample - use builder API
        # Store scale factor for dynamic upsample creation during forward pass
        self.final_upsample_scale = (
            common_stride if isinstance(common_stride, tuple) else (common_stride, common_stride)
        )
        self.final_upsample_mode = "nearest"
        logger.debug("TtPanopticDeepLabSemSegHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, Dict]:
        """
        The forward pass for the Panoptic head in inference mode.
        """
        logger.debug("TtPanopticDeepLabSemSegHead forward pass starting")
        y = self.layers(features)

        # Create dynamic upsample for final output
        from models.tt_cnn.tt.builder import UpsampleConfiguration

        final_upsample_config = UpsampleConfiguration(
            scale_factor=self.final_upsample_scale,
            mode=self.final_upsample_mode,
            input_height=y.shape[1],
            input_width=y.shape[2],
            batch_size=y.shape[0],
            channels=y.shape[3],
        )
        final_upsample = TtUpsample(final_upsample_config, self.device)
        y = final_upsample(y)

        logger.debug(f"TtPanopticDeepLabSemSegHead forward pass complete - final output shape: {y.shape}")
        return y, {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        y = super().layers(features)

        y = self.head_0(y)
        y = self.activation(y)

        y = self.head_1(y)
        y = self.activation(y)

        y = self.predictor(y)
        return y
