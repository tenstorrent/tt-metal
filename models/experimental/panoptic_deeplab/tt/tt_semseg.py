# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Union, Optional, Tuple
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_aspp import TtASPP
from models.experimental.panoptic_deeplab.tt.tt_upsample import BilinearUpsampleMatmulTTNN as TtBilinearUpsample
from models.tt_cnn.tt.builder import TtConv2d
from models.experimental.panoptic_deeplab.reference.pytorch_semseg import ShapeSpec
from models.common.lightweightmodule import LightweightModule
from models.experimental.panoptic_deeplab.tt.common import reshape_flattened_conv_output


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

        # Initialize decoder with all stages
        self._initialize_decoder(
            parameters,
            in_channels,
            in_strides,
            project_channels,
            decoder_channels,
            aspp_dilations,
            aspp_dropout,
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
        train_size,
    ):
        """Initialize decoder stages"""
        self.decoder = {}

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
        logger.info(f"🔷 Executing conv: decoder.{aspp_feature_key}.project_conv (ASPP)")
        y = stage["project_conv"](x)
        # Reshape logic for decoder.res5.project_conv (ASPP) - ASPP already handles its own reshape internally
        # But add safety check in case it's flattened
        y = reshape_flattened_conv_output(y, batch_size=1, layer_name=f"decoder.{aspp_feature_key}.project_conv")
        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtDeepLabV3PlusHead ASPP stage complete, output shape: {y.shape}")

        # --- Subsequent Fusion Stages (e.g., 'res3', then 'res2') ---
        for i, f_key in enumerate(feature_keys[1:]):
            logger.debug(f"TtDeepLabV3PlusHead processing fusion stage {i+1} with feature: {f_key}")
            previous_y = y
            x = features[f_key]
            stage = self.decoder[f_key]
            logger.info(f"🔷 Executing conv: decoder.{f_key}.project_conv")
            proj_x = stage["project_conv"](x)
            # Reshape logic for decoder.{res2,res3,res4}.project_conv that now outputs flattened format [1,1,NHW,C] -> [N,H,W,C]
            proj_x = reshape_flattened_conv_output(proj_x, batch_size=1, layer_name=f"decoder.{f_key}.project_conv")
            proj_x = ttnn.to_memory_config(proj_x, ttnn.DRAM_MEMORY_CONFIG)
            logger.debug(f"TtDeepLabV3PlusHead fusion stage {i+1} projection complete, shape: {proj_x.shape}")

            # Save y shape before layout conversion (layout conversion may flatten the tensor)
            y_height, y_width = y.shape[1], y.shape[2]
            y_batch, y_channels = y.shape[0], y.shape[3]

            scale_h = proj_x.shape[1] // y_height
            scale_w = proj_x.shape[2] // y_width

            logger.info(
                f"Decoder upsample stage {i+1} ({f_key}): y_shape=[{y_batch}, {y_height}, {y_width}, {y_channels}], proj_x_shape={proj_x.shape}, scale_factors=({scale_h}, {scale_w})"
            )
            logger.info(f"  Target size from proj_x: H={proj_x.shape[1]}, W={proj_x.shape[2]}")
            logger.info(f"  Calculated output size: H={y_height * scale_h}, W={y_width * scale_w}")
            if y_height * scale_h != proj_x.shape[1] or y_width * scale_w != proj_x.shape[2]:
                logger.warning(
                    f"  Size mismatch! Expected [{proj_x.shape[1]}, {proj_x.shape[2]}] but will get [{y_height * scale_h}, {y_width * scale_w}]"
                )

            # Use TtUpsample wrapper with channel slicing
            from models.tt_cnn.tt.builder import TtUpsample, UpsampleConfiguration

            # Convert to interleaved DRAM if sharded
            if y.is_sharded():
                y = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
            else:
                y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

            # Convert to ROW_MAJOR for bilinear upsampling
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

            # Reshape if flattened (to_layout may flatten to [N, 1, H*W, C])
            if y.shape[1] == 1 and y.shape[2] == y_height * y_width:
                y = ttnn.reshape(y, (y_batch, y_height, y_width, y_channels))

            logger.debug(
                f"Decoder upsample stage {i+1}: input shape={y.shape}, dtype={y.dtype}, layout={y.layout}, scale_factors=({scale_h}, {scale_w})"
            )

            # Create dynamic upsample configuration with channel slicing
            upsample_config = UpsampleConfiguration(
                input_height=y_height,
                input_width=y_width,
                channels=y_channels,
                batch_size=y_batch,
                scale_factor=(scale_h, scale_w),
                mode="nearest",  # accuracy changes are negligible, so we use nearest for performance
                slice_strategy=None,
            )
            upsample_layer = TtUpsample(upsample_config, self.device)
            y_upsampled = upsample_layer(y)

            # Convert back to TILE_LAYOUT and DRAM
            y_upsampled = ttnn.to_layout(y_upsampled, ttnn.TILE_LAYOUT)
            y_upsampled = ttnn.to_memory_config(y_upsampled, ttnn.DRAM_MEMORY_CONFIG)

            # Ensure both tensors have the same dtype before concatenation
            target_dtype = ttnn.bfloat8_b
            if proj_x.dtype != target_dtype:
                proj_x = ttnn.typecast(proj_x, target_dtype)
            if y_upsampled.dtype != target_dtype:
                y_upsampled = ttnn.typecast(y_upsampled, target_dtype)

            y = ttnn.concat([proj_x, y_upsampled], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Save shape before deallocation for potential reshape later
            expected_H, expected_W = proj_x.shape[1], proj_x.shape[2]

            ttnn.deallocate(previous_y)
            ttnn.deallocate(proj_x)
            ttnn.deallocate(y_upsampled)
            # Use single conv versions - slicing handled by config, ReLU is fused
            logger.info(f"🔷 Executing conv: decoder.{f_key}.fuse_conv.0")
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
            y_conv0 = stage["fuse_conv_0"](y)
            ttnn.deallocate(y)

            is_flattened = y_conv0.shape[1] == 1 and y_conv0.shape[2] == expected_H * expected_W
            if is_flattened:
                N = y_conv0.shape[0]
                C = y_conv0.shape[3]
                y_conv0 = ttnn.reshape(y_conv0, (N, expected_H, expected_W, C))
                logger.debug(
                    f"Reshaped channel-sliced fuse_conv_0 output from [1, 1, {expected_H * expected_W}, {C}] to [{N}, {expected_H}, {expected_W}, {C}]"
                )

            # Only apply manual ReLU for channel-sliced fuse_conv.0 (res3, res2)
            if f_key in ["res3", "res2"]:
                y_conv0 = ttnn.relu(y_conv0)

            ttnn.to_memory_config(y_conv0, ttnn.DRAM_MEMORY_CONFIG)
            logger.info(f"🔷 Executing conv: decoder.{f_key}.fuse_conv.1")
            y_conv1 = stage["fuse_conv_1"](y_conv0)
            ttnn.deallocate(y_conv0)

            is_flattened_conv1 = y_conv1.shape[1] == 1 and y_conv1.shape[2] == expected_H * expected_W
            if is_flattened_conv1:
                N = y_conv1.shape[0]
                C = y_conv1.shape[3]
                y_conv1 = ttnn.reshape(y_conv1, (N, expected_H, expected_W, C))
                logger.debug(
                    f"Reshaped height-sliced fuse_conv_1 output from [1, 1, {expected_H * expected_W}, {C}] to [{N}, {expected_H}, {expected_W}, {C}]"
                )

            # fuse_conv.1 uses width slicing, so fused ReLU is used
            y = y_conv1

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
        decoder_out_ch = decoder_channels[0]
        logger.debug(f"Initializing TtPanopticDeepLabSemSegHead with {num_classes} classes")

        # Head 0 - use builder API
        head_params = parameters["head"] if isinstance(parameters, dict) else parameters.head
        self.head_0 = self._create_conv_layer(head_params[0], "semantic_head.head.0")
        head_0_config = (
            head_params[0]["conv_config"] if isinstance(head_params[0], dict) else head_params[0].conv_config
        )
        logger.debug(
            f"head.0 config: stride={head_0_config.stride}, kernel={head_0_config.kernel_size}, padding={head_0_config.padding}"
        )

        # Head 1 - use builder API
        self.head_1 = self._create_conv_layer(head_params[1], "semantic_head.head.1")
        head_1_config = (
            head_params[1]["conv_config"] if isinstance(head_params[1], dict) else head_params[1].conv_config
        )
        logger.debug(
            f"head.1 config: stride={head_1_config.stride}, kernel={head_1_config.kernel_size}, padding={head_1_config.padding}"
        )

        # Predictor - use builder API
        predictor_params = parameters["predictor"] if isinstance(parameters, dict) else parameters.predictor
        self.predictor = self._create_conv_layer(predictor_params, "semantic_head.predictor")

        # Store original output channels if predictor was padded (for slicing back)
        self.predictor_original_out_channels = predictor_params.get("original_out_channels", None)

        # Track original output channels for torch conversion (will be set during forward if needed)
        self._last_output_original_channels = None

        # Final upsample - matmul based bilinear upsample
        # perf wise this makes sense because we dont need to permute to channel last after it
        # we dont need to permute to channel last because this is final output that goes to the host
        # todo: remove hardcoded values; they should come from infer_ttnn_params in the preprocessing step

        # First config
        final_upsample_mm_config1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(5, 4),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=2,
            out_block_h=4,
            out_block_w=2,
            per_core_M=4,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            num_global_cb_receivers=0,
            untilize_out=False,
        )

        # Second config
        final_upsample_mm_config2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(5, 4),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=2,
            out_block_h=16,
            out_block_w=2,
            per_core_M=16,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            num_global_cb_receivers=0,
            untilize_out=False,
        )

        self.final_upsample = TtBilinearUpsample(
            device,
            input_batch=1,
            input_channels=32,  # true value is 19; this is padded somewhere
            input_height=128,
            input_width=256,
            scale=common_stride,
            input_channels_first=False,
            output_channels_first=True,
            mm1_program_config=final_upsample_mm_config1,
            mm2_program_config=final_upsample_mm_config2,
            output_dtype=ttnn.bfloat8_b,
        )
        logger.debug("TtPanopticDeepLabSemSegHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, Dict]:
        """
        The forward pass for the Panoptic head in inference mode.
        """
        logger.debug("TtPanopticDeepLabSemSegHead forward pass starting")

        # Save spatial dimensions from decoder output before calling layers()
        # (layers may flatten during convs, but we need original dims for reshape/upsample)
        # Use first feature in in_features list
        decoder_out = features[self.in_features[0]]
        self._saved_spatial_h = decoder_out.shape[1]
        self._saved_spatial_w = decoder_out.shape[2]
        logger.debug(f"Saved spatial dimensions from decoder: H={self._saved_spatial_h}, W={self._saved_spatial_w}")

        y = self.layers(features)

        # Handle final upsample - use direct ttnn.upsample call (wrapper flattens spatial dims)
        logger.debug(f"Before final upsample: shape={y.shape}, layout={y.layout}")

        # Use actual head spatial dimensions (saved in layers() method)
        current_h = self._actual_head_h
        current_w = self._actual_head_w

        # Force reshape if dimensions don't match expected [N, H, W, C] format
        if y.shape[1] != current_h or y.shape[2] != current_w:
            logger.debug(f"Reshaping output from {y.shape} to [1, {current_h}, {current_w}, {y.shape[3]}]")
            y = ttnn.reshape(y, (y.shape[0], current_h, current_w, y.shape[3]))
            logger.debug(f"After reshape: {y.shape}")

        # Convert to interleaved DRAM if sharded
        if y.is_sharded():
            y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        else:
            y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)

        # Convert to ROW_MAJOR for upsample
        # Staying tilized causes OOM during upsample;
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

        # Check allocation before final upsample
        if not y.is_allocated():
            logger.warning(f"Final upsample: input y is NOT allocated before upsample!")
        else:
            logger.debug(f"Final upsample: input y is allocated (shape={y.shape}, dtype={y.dtype}, layout={y.layout})")

        # Matmul based upsample
        logger.debug(f"y shape: {y.shape}")
        y = ttnn.reshape(y, (y.shape[0], 128, 256, y.shape[3]))
        y = self.final_upsample(y)

        # Check allocation after final upsample
        if not y.is_allocated():
            logger.warning(f"Final upsample: output y is NOT allocated after upsample!")
        else:
            logger.debug(f"Final upsample: output y is allocated (shape={y.shape}, dtype={y.dtype}, layout={y.layout})")

        logger.debug(f"After final upsample: shape={y.shape}, layout={y.layout}")

        # Store original channel count for torch-side slicing if predictor was padded
        if self.predictor_original_out_channels is not None:
            self._last_output_original_channels = self.predictor_original_out_channels
            logger.debug(
                f"Final output at {y.shape[3]} channels, will slice to {self.predictor_original_out_channels} in torch"
            )

        logger.debug(f"TtPanopticDeepLabSemSegHead forward pass complete - final output shape: {y.shape}")
        return y, {}

    def get_output_channels_for_slicing(self):
        """
        Get the original output channels if predictor was padded.
        Returns None if no padding was applied (i.e., output is already correct size).
        """
        return self._last_output_original_channels

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        y = super().layers(features)

        # Save spatial dimensions right after super().layers() before any head convs
        # (these are the actual dimensions for this head, not decoder output)
        self._actual_head_h = y.shape[1]
        self._actual_head_w = y.shape[2]
        logger.debug(f"Actual head input dimensions: H={self._actual_head_h}, W={self._actual_head_w}")

        logger.info(f"🔷 Executing conv: semantic_head.head.0")
        y = self.head_0(y)

        if y.shape[1] == 1 and y.shape[2] == self._actual_head_h * self._actual_head_w:
            y = ttnn.reshape(y, (y.shape[0], self._actual_head_h, self._actual_head_w, y.shape[3]))

        logger.info(f"🔷 Executing conv: semantic_head.head.1")
        y = self.head_1(y)

        if y.shape[1] == 1 and y.shape[2] == self._actual_head_h * self._actual_head_w:
            y = ttnn.reshape(y, (y.shape[0], self._actual_head_h, self._actual_head_w, y.shape[3]))

        # Convert to interleaved for predictor
        if y.is_sharded():
            y = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
        else:
            y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

        logger.info(f"🔷 Executing conv: semantic_head.predictor")
        y = self.predictor(y)

        return y
