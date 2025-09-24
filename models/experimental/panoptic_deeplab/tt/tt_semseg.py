# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Union, Optional, Tuple
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_aspp import TtASPP, get_ttnn_activation

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import TtConv2d, TtConv2dParameters
from models.experimental.panoptic_deeplab.tt.tt_upsample_wrapper import TtUpsample
from models.experimental.panoptic_deeplab.reference.pytorch_semseg import ShapeSpec
from models.common.lightweightmodule import LightweightModule


class TtDeepLabV3PlusHead(LightweightModule):
    """
    TTNN implementation of the DeepLabV3+ segmentation head.
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
        self.activation = get_ttnn_activation("relu")

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
                )
                decoder_stage["project_conv"] = project_conv
                decoder_stage["fuse_conv_0"] = None
            else:
                proj_out_ch = project_channels[idx]
                fuse_out_ch = decoder_channels[idx]

                # Parameter path for this decoder stage
                base_path = parameters[feature_name]

                # Project Conv
                # Handle both dict and object parameter formats
                proj_conv_path = base_path["project_conv"] if isinstance(base_path, dict) else base_path.project_conv
                proj_bias = (
                    proj_conv_path["bias"]
                    if isinstance(proj_conv_path, dict) and "bias" in proj_conv_path
                    else (proj_conv_path.bias if hasattr(proj_conv_path, "bias") else None)
                )
                proj_weight = proj_conv_path["weight"] if isinstance(proj_conv_path, dict) else proj_conv_path.weight
                proj_params = TtConv2dParameters(weight=proj_weight, bias=proj_bias, device=self.device)
                project_conv = TtConv2d.create(proj_params, stride=(1, 1), padding=(0, 0))

                # Fuse Conv 0
                fuse_conv_params = base_path["fuse_conv"] if isinstance(base_path, dict) else base_path.fuse_conv
                fuse0_path = fuse_conv_params[0]
                fuse0_bias = (
                    fuse0_path["bias"]
                    if isinstance(fuse0_path, dict) and "bias" in fuse0_path
                    else (fuse0_path.bias if hasattr(fuse0_path, "bias") else None)
                )
                fuse0_weight = fuse0_path["weight"] if isinstance(fuse0_path, dict) else fuse0_path.weight
                fuse0_params = TtConv2dParameters(weight=fuse0_weight, bias=fuse0_bias, device=self.device)

                fuse_conv_0_no_slice = TtConv2d.create(fuse0_params, stride=(1, 1), padding=(1, 1))
                if fuse0_params.in_channels == 160:
                    fuse_conv_0_slice = TtConv2d.create_with_channel_slicing(
                        fuse0_params, num_slices=5, stride=(1, 1), padding=(1, 1)
                    )
                else:
                    fuse_conv_0_slice = TtConv2d.create_with_height_slicing(
                        fuse0_params, num_slices=4, stride=(1, 1), padding=(1, 1)
                    )

                # Fuse Conv 1
                fuse1_path = fuse_conv_params[1]
                fuse1_bias = (
                    fuse1_path["bias"]
                    if isinstance(fuse1_path, dict) and "bias" in fuse1_path
                    else (fuse1_path.bias if hasattr(fuse1_path, "bias") else None)
                )
                fuse1_weight = fuse1_path["weight"] if isinstance(fuse1_path, dict) else fuse1_path.weight
                fuse1_params = TtConv2dParameters(weight=fuse1_weight, bias=fuse1_bias, device=self.device)

                fuse_conv_1_no_slice = TtConv2d.create(fuse1_params, stride=(1, 1), padding=(1, 1))
                fuse_conv_1_height_slice = TtConv2d.create_with_height_slicing(
                    fuse1_params, num_slices=2, stride=(1, 1), padding=(1, 1)
                )

                decoder_stage["project_conv"] = project_conv
                decoder_stage["fuse_conv_0_no_slice"] = fuse_conv_0_no_slice
                decoder_stage["fuse_conv_0_height_slice"] = fuse_conv_0_slice
                decoder_stage["fuse_conv_1_no_slice"] = fuse_conv_1_no_slice
                decoder_stage["fuse_conv_1_height_slice"] = fuse_conv_1_height_slice

            self.decoder[feature_name] = decoder_stage

        # Initialize upsample operations for decoder
        self.upsample_standard = TtUpsample.create(device=device, scale_factor=(1, 1), mode="bilinear")
        self.upsample_with_slicing = TtUpsample.create_with_channel_slicing(
            device=device, scale_factor=(1, 1), mode="bilinear", num_slices=2
        )
        logger.debug("TtDeepLabV3PlusHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Union[ttnn.Tensor, Tuple[ttnn.Tensor, Dict]]:
        y = self.layers(features)
        return y

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """
        Executes the decoder pipeline, mirroring the PyTorch version's logic.
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

            # Use appropriate upsample operation based on iteration
            if i == 0:
                # No slicing for first iteration
                self.upsample_standard._scale_factor = (scale_h, scale_w)
                y_upsampled = self.upsample_standard(y)
            else:
                # Channel slicing for subsequent iterations
                self.upsample_with_slicing._scale_factor = (scale_h, scale_w)
                y_upsampled = self.upsample_with_slicing(y)

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

            # Choose the appropriate conv based on iteration index
            if i == 0:
                y_conv0 = stage["fuse_conv_0_no_slice"](y)
            else:
                y_conv0 = stage["fuse_conv_0_height_slice"](y)
            ttnn.deallocate(y)
            y_act0 = self.activation(y_conv0)
            ttnn.deallocate(y_conv0)

            ttnn.to_memory_config(y_act0, ttnn.DRAM_MEMORY_CONFIG)
            # Choose the appropriate conv based on iteration index
            if i == 0:
                y_conv1 = stage["fuse_conv_1_no_slice"](y_act0)
            else:
                y_conv1 = stage["fuse_conv_1_height_slice"](y_act0)
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
        )
        assert self.decoder_only
        use_bias = norm == ""
        decoder_out_ch = decoder_channels[0]
        logger.debug(f"Initializing TtPanopticDeepLabSemSegHead with {num_classes} classes")

        # Head 0
        # Handle both dict and object parameter formats
        head_params = parameters["head"] if isinstance(parameters, dict) else parameters.head
        head0_path = head_params[0]
        head0_bias = (
            head0_path["bias"]
            if isinstance(head0_path, dict) and "bias" in head0_path
            else (head0_path.bias if hasattr(head0_path, "bias") else None)
        )
        head0_weight = head0_path["weight"] if isinstance(head0_path, dict) else head0_path.weight
        head0_params = TtConv2dParameters(
            weight=head0_weight,
            bias=head0_bias,
            device=self.device,
        )
        self.head_0 = TtConv2d.create_with_height_slicing(head0_params, num_slices=2, stride=(1, 1), padding=(1, 1))

        # Head 1
        head1_path = head_params[1]
        head1_bias = (
            head1_path["bias"]
            if isinstance(head1_path, dict) and "bias" in head1_path
            else (head1_path.bias if hasattr(head1_path, "bias") else None)
        )
        head1_weight = head1_path["weight"] if isinstance(head1_path, dict) else head1_path.weight
        head1_params = TtConv2dParameters(
            weight=head1_weight,
            bias=head1_bias,
            device=self.device,
        )
        self.head_1 = TtConv2d.create_with_height_slicing(head1_params, num_slices=2, stride=(1, 1), padding=(1, 1))

        # Predictor
        predictor_params = parameters["predictor"] if isinstance(parameters, dict) else parameters.predictor
        predictor_path = predictor_params
        predictor_bias = (
            predictor_path["bias"]
            if isinstance(predictor_path, dict) and "bias" in predictor_path
            else (predictor_path.bias if hasattr(predictor_path, "bias") else None)
        )
        predictor_weight = predictor_path["weight"] if isinstance(predictor_path, dict) else predictor_path.weight
        predictor_params = TtConv2dParameters(
            weight=predictor_weight,
            bias=predictor_bias,
            device=self.device,
        )
        self.predictor = TtConv2d.create(predictor_params, stride=(1, 1), padding=(0, 0))

        self.final_upsample = TtUpsample.create(device=device, scale_factor=common_stride, mode="nearest")
        logger.debug("TtPanopticDeepLabSemSegHead initialization complete")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, Dict]:
        """
        The forward pass for the Panoptic head in inference mode.
        """
        logger.debug("TtPanopticDeepLabSemSegHead forward pass starting")
        y = self.layers(features)

        y = self.final_upsample(y)
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
