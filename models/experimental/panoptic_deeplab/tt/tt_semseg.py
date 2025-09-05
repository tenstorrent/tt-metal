import torch
from typing import Dict, List, Union, Optional, Tuple
from torch import nn
import ttnn

from .tt_aspp import TtASPP, get_ttnn_norm, get_ttnn_activation

from .tt_conv2dWrapper import TtConv2d, TtConv2dParameters
from .tt_upsample_wrapper import TtUpsample
from ..reference.pytorch_semSeg import ShapeSpec


class TtDeepLabV3PlusHead(nn.Module):
    """
    TTNN implementation of the DeepLabV3+ segmentation head.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        device: ttnn.Device,
        *,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: str,
        train_size: Optional[Tuple],
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        predictor_weight: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        sorted_input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted_input_shape]
        in_channels = [v.channels for k, v in sorted_input_shape]
        in_strides = [v.stride for k, v in sorted_input_shape]
        aspp_channels = decoder_channels[-1]

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

                project_conv = TtASPP(
                    in_channels=in_channel,
                    out_channels=aspp_channels,
                    dilations=aspp_dilations,
                    device=self.device,
                    norm=norm,
                    activation="relu",
                    dropout=aspp_dropout,
                    pool_kernel_size=pool_kernel_size,
                    shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
                    shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
                    shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
                )
                decoder_stage["project_conv"] = project_conv
                decoder_stage["fuse_conv_0"] = None
            else:
                proj_out_ch = project_channels[idx]
                param_dict = {"weight": project_conv_weights[feature_name]}
                if use_bias:
                    param_dict["bias"] = torch.zeros(1, 1, 1, proj_out_ch)
                parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
                project_conv = TtConv2d.create(parameters, stride=(1, 1), padding=(0, 0))

                fuse_in_ch = proj_out_ch + decoder_channels[idx + 1]
                fuse_out_ch = decoder_channels[idx]
                # Create fuse_conv_0 with no slicing (for i=0) and height slicing (for i>0)
                param_dict = {"weight": fuse_conv_0_weights[feature_name]}
                if use_bias:
                    param_dict["bias"] = torch.zeros(1, 1, 1, fuse_out_ch)
                parameters_no_slice = TtConv2dParameters.from_torch(param_dict, device=self.device)
                fuse_conv_0_no_slice = TtConv2d.create(parameters_no_slice, stride=(1, 1), padding=(1, 1))

                # Create height-sliced version for i>0
                param_dict = {"weight": fuse_conv_0_weights[feature_name]}
                if use_bias:
                    param_dict["bias"] = torch.zeros(1, 1, 1, fuse_out_ch)
                parameters_height_slice = TtConv2dParameters.from_torch(param_dict, device=self.device)
                fuse_conv_0_height_slice = TtConv2d.create_with_height_slicing(
                    parameters_height_slice, num_slices=4, stride=(1, 1), padding=(1, 1)
                )

                # Create fuse_conv_1 with no slicing (for i=0) and height slicing (for i>0)
                param_dict = {"weight": fuse_conv_1_weights[feature_name]}
                if use_bias:
                    param_dict["bias"] = torch.zeros(1, 1, 1, fuse_out_ch)
                parameters_no_slice = TtConv2dParameters.from_torch(param_dict, device=self.device)
                fuse_conv_1_no_slice = TtConv2d.create(parameters_no_slice, stride=(1, 1), padding=(1, 1))

                # Create height-sliced version for i>0
                param_dict = {"weight": fuse_conv_1_weights[feature_name]}
                if use_bias:
                    param_dict["bias"] = torch.zeros(1, 1, 1, fuse_out_ch)
                parameters_height_slice = TtConv2dParameters.from_torch(param_dict, device=self.device)
                fuse_conv_1_height_slice = TtConv2d.create_with_height_slicing(
                    parameters_height_slice, num_slices=2, stride=(1, 1), padding=(1, 1)
                )

                decoder_stage["project_conv"] = project_conv
                decoder_stage["project_norm"] = get_ttnn_norm(norm, proj_out_ch, device, norm_params=None)
                decoder_stage["fuse_conv_0_no_slice"] = fuse_conv_0_no_slice
                decoder_stage["fuse_conv_0_height_slice"] = fuse_conv_0_height_slice
                decoder_stage["fuse_norm_0"] = get_ttnn_norm(norm, fuse_out_ch, device, norm_params=None)
                decoder_stage["fuse_conv_1_no_slice"] = fuse_conv_1_no_slice
                decoder_stage["fuse_conv_1_height_slice"] = fuse_conv_1_height_slice
                decoder_stage["fuse_norm_1"] = get_ttnn_norm(norm, fuse_out_ch, device, norm_params=None)

            self.decoder[feature_name] = decoder_stage

        # Initialize upsample operations for decoder
        self.upsample_standard = TtUpsample.create(device=device, scale_factor=(1, 1), mode="bilinear")
        self.upsample_with_slicing = TtUpsample.create_with_channel_slicing(
            device=device, scale_factor=(1, 1), mode="bilinear", num_slices=2
        )

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Union[ttnn.Tensor, Tuple[ttnn.Tensor, Dict]]:
        y = self.layers(features)
        return y

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """
        Executes the decoder pipeline, mirroring the PyTorch version's logic.
        """
        y = None
        feature_keys = self.in_features[::-1]

        # --- Stage 1: ASPP ---
        aspp_feature_key = feature_keys[0]
        x = features[aspp_feature_key]
        stage = self.decoder[aspp_feature_key]
        y = stage["project_conv"](x)
        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

        # --- Subsequent Fusion Stages (e.g., 'res3', then 'res2') ---
        for i, f_key in enumerate(feature_keys[1:]):
            previous_y = y
            x = features[f_key]
            stage = self.decoder[f_key]
            proj_x = stage["project_conv"](x)
            proj_x = stage["project_norm"](proj_x)
            proj_x = self.activation(proj_x)
            proj_x = ttnn.to_memory_config(proj_x, ttnn.DRAM_MEMORY_CONFIG)

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
            y_norm0 = stage["fuse_norm_0"](y_conv0)
            ttnn.deallocate(y_conv0)

            y_act0 = self.activation(y_norm0)
            ttnn.deallocate(y_norm0)

            ttnn.to_memory_config(y_act0, ttnn.DRAM_MEMORY_CONFIG)
            # Choose the appropriate conv based on iteration index
            if i == 0:
                y_conv1 = stage["fuse_conv_1_no_slice"](y_act0)
            else:
                y_conv1 = stage["fuse_conv_1_height_slice"](y_act0)
            y_norm1 = stage["fuse_norm_1"](y_conv1)
            ttnn.deallocate(y_conv1)

            y = self.activation(y_norm1)
            ttnn.deallocate(y_norm1)

        return y


class TtPanopticDeepLabSemSegHead(TtDeepLabV3PlusHead):
    """
    TTNN implementation of the Panoptic-DeepLab semantic segmentation head.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        device: ttnn.Device,
        *,
        head_channels: int,
        num_classes: int,
        norm: str,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        train_size: Optional[Tuple],
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        panoptic_head_0_weight: torch.Tensor,
        panoptic_head_1_weight: torch.Tensor,
        panoptic_predictor_weight: torch.Tensor,
    ):
        super().__init__(
            input_shape=input_shape,
            device=device,
            norm=norm,
            num_classes=None,
            predictor_weight=None,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            train_size=train_size,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            project_conv_weights=project_conv_weights,
            fuse_conv_0_weights=fuse_conv_0_weights,
            fuse_conv_1_weights=fuse_conv_1_weights,
        )
        assert self.decoder_only
        use_bias = norm == ""
        decoder_out_ch = decoder_channels[0]

        # Create head_0 with height slicing
        param_dict = {"weight": panoptic_head_0_weight}
        if use_bias:
            param_dict["bias"] = torch.zeros(1, 1, 1, decoder_out_ch)
        parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
        self.head_0 = TtConv2d.create_with_height_slicing(parameters, num_slices=2, stride=(1, 1), padding=(1, 1))
        self.head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=None)

        # Create head_1 with height slicing
        param_dict = {"weight": panoptic_head_1_weight}
        if use_bias:
            param_dict["bias"] = torch.zeros(1, 1, 1, head_channels)
        parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
        self.head_1 = TtConv2d.create_with_height_slicing(parameters, num_slices=2, stride=(1, 1), padding=(1, 1))
        self.head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=None)

        # Create predictor without slicing
        param_dict = {"weight": panoptic_predictor_weight, "bias": torch.zeros(1, 1, 1, num_classes)}
        parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
        self.predictor = TtConv2d.create(parameters, stride=(1, 1), padding=(0, 0))

        # Initialize final upsample for semantic head
        # Use default mode if bilinear causes memory issues, otherwise bilinear for PCC
        self.final_upsample = TtUpsample.create(device=device, scale_factor=common_stride, mode="nearest")

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, Dict]:
        """
        The forward pass for the Panoptic head in inference mode.
        """
        y = self.layers(features)

        y = self.final_upsample(y)
        return y, {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        y = super().layers(features)

        y = self.head_0(y)
        y = self.head_norm_0(y)
        y = self.activation(y)

        y = self.head_1(y)
        y = self.head_norm_1(y)
        y = self.activation(y)

        y = self.predictor(y)
        return y
