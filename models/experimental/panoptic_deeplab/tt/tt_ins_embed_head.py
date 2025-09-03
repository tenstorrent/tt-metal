import torch
from typing import Dict, List, Optional, Tuple
from torch import nn
import ttnn

from .tt_aspp import TtASPP, get_ttnn_norm, get_ttnn_activation
from .tt_conv2dWrapper import TtConv2d, TtConv2dParameters
from .tt_pytorch_semSeg import ShapeSpec


class TtDeepLabV3PlusInsEmbedHead(nn.Module):
    """
    TTNN implementation of the DeepLabV3+ instance embedding head base class.
    Similar to semantic segmentation head but used for instance embedding prediction.
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
        # --- All weights must be provided externally ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        sorted_input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted_input_shape]
        in_channels = [v.channels for k, v in sorted_input_shape]
        in_strides = [v.stride for k, v in sorted_input_shape]
        aspp_channels = decoder_channels[-1]

        self.common_stride = common_stride
        self.device = device
        self.activation = get_ttnn_activation("relu")

        def _create_tt_conv2d(
            weight: torch.Tensor,
            in_ch: int,
            out_ch: int,
            kernel_size: int,
            stride: int,
            padding: int,
            use_bias: bool,
            slice_num: int = 1,
            toSlice: bool = False,
        ):
            param_dict = {"weight": weight}
            param_dict["channel_slice_num"] = slice_num
            if use_bias:
                param_dict["bias"] = torch.zeros(1, 1, 1, out_ch)

            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
            return TtConv2d(parameters, stride=(stride, stride), padding=(padding, padding), toSlice=toSlice)

        self.decoder = {}
        use_bias = norm == ""

        for idx, in_channel in enumerate(in_channels):
            decoder_stage = {}
            feature_name = self.in_features[idx]

            if idx == len(self.in_features) - 1:  # ASPP stage
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
            else:  # Low-level feature stages
                proj_out_ch = project_channels[idx]
                project_conv = _create_tt_conv2d(
                    weight=project_conv_weights[feature_name],
                    in_ch=in_channel,
                    out_ch=proj_out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_bias=use_bias,
                )

                fuse_in_ch = proj_out_ch + decoder_channels[idx + 1]
                fuse_out_ch = decoder_channels[idx]
                fuse_conv_0 = _create_tt_conv2d(
                    weight=fuse_conv_0_weights[feature_name],
                    in_ch=fuse_in_ch,
                    out_ch=fuse_out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=use_bias,
                    slice_num=3,
                )
                fuse_conv_1 = _create_tt_conv2d(
                    weight=fuse_conv_1_weights[feature_name],
                    in_ch=fuse_out_ch,
                    out_ch=fuse_out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=use_bias,
                    slice_num=3,
                )

                decoder_stage["project_conv"] = project_conv
                decoder_stage["project_norm"] = get_ttnn_norm(norm, proj_out_ch, device)
                decoder_stage["fuse_conv_0"] = fuse_conv_0
                decoder_stage["fuse_norm_0"] = get_ttnn_norm(norm, fuse_out_ch, device)
                decoder_stage["fuse_conv_1"] = fuse_conv_1
                decoder_stage["fuse_norm_1"] = get_ttnn_norm(norm, fuse_out_ch, device)

            self.decoder[feature_name] = decoder_stage

    def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """
        Executes the decoder pipeline for instance embedding head.
        """
        y = None
        feature_keys = self.in_features[::-1]

        # --- Stage 1: ASPP (on the coarsest feature map, e.g., 'res5') ---
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

            # 1. Project the low-level features with a 1x1 Conv, followed by Norm and Activation.
            proj_x = stage["project_conv"](x)
            proj_x = stage["project_norm"](proj_x)
            proj_x = self.activation(proj_x)
            proj_x = ttnn.to_memory_config(proj_x, ttnn.DRAM_MEMORY_CONFIG)

            scale_h = proj_x.shape[1] // y.shape[1]
            scale_w = proj_x.shape[2] // y.shape[2]
            y_upsampled = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

            # Get original dimensions before flattening
            orig_batch, orig_height, orig_width, orig_channels = y_upsampled.shape

            # Channel slicing for upsample operation - always slice by 2 (except first iteration)
            if i == 0:
                # First iteration - no slicing, use original upsample pattern
                y_upsampled = ttnn.upsample(y_upsampled, scale_factor=(scale_h, scale_w), mode="bilinear")
            else:
                # All other iterations - always slice by 2
                split_factor = 2
                channels_per_slice = orig_channels // split_factor
                print(f"Using channel slicing for upsample: {orig_channels} channels split into {split_factor} slices")
                sliced_results = []
                for slice_idx in range(split_factor):
                    start_ch = slice_idx * channels_per_slice
                    end_ch = (slice_idx + 1) * channels_per_slice

                    # Slice the tensor along channel dimension (NHWC format, channel is dim 3)
                    y_slice = ttnn.slice(
                        y_upsampled, [0, 0, 0, start_ch], [orig_batch, orig_height, orig_width, end_ch]
                    )

                    # Upsample the slice using original pattern
                    y_slice_upsampled = ttnn.upsample(y_slice, scale_factor=(scale_h, scale_w), mode="bilinear")
                    y_slice_upsampled = ttnn.to_memory_config(y_slice_upsampled, ttnn.DRAM_MEMORY_CONFIG)

                    sliced_results.append(y_slice_upsampled)
                    ttnn.deallocate(y_slice)
                y_upsampled = ttnn.concat(sliced_results, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for slice_result in sliced_results:
                    ttnn.deallocate(slice_result)

            y_upsampled = ttnn.to_memory_config(y_upsampled, ttnn.DRAM_MEMORY_CONFIG)
            y_upsampled = ttnn.to_layout(y_upsampled, ttnn.TILE_LAYOUT)

            # 3. Fuse the features by concatenating along the channel dimension (dim=3 for NHWC).
            y = ttnn.concat([proj_x, y_upsampled], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            ttnn.deallocate(previous_y)
            ttnn.deallocate(proj_x)
            ttnn.deallocate(y_upsampled)

            if i == 0:
                y_conv0 = stage["fuse_conv_0"](y)
            else:
                y_conv0 = stage["fuse_conv_0"](
                    y,
                    slice_config=ttnn.Conv2dSliceConfig(
                        slice_type=ttnn.Conv2dSliceHeight,
                        num_slices=4,
                    ),
                )
            ttnn.deallocate(y)
            y_norm0 = stage["fuse_norm_0"](y_conv0)
            # can't do this for some reason
            # ttnn.deallocate(y_conv0)

            y_act0 = self.activation(y_norm0)
            ttnn.deallocate(y_norm0)

            ttnn.to_memory_config(y_act0, ttnn.DRAM_MEMORY_CONFIG)
            if i == 0:
                y_conv1 = stage["fuse_conv_1"](y_act0)
            else:
                y_conv1 = stage["fuse_conv_1"](
                    y_act0,
                    slice_config=ttnn.Conv2dSliceConfig(
                        slice_type=ttnn.Conv2dSliceHeight,
                        num_slices=2,
                    ),
                )

            y_norm1 = stage["fuse_norm_1"](y_conv1)
            # can't do this for some reason
            # ttnn.deallocate(y_conv1)

            y = self.activation(y_norm1)
            ttnn.deallocate(y_norm1)

        return y


class TtPanopticDeepLabInsEmbedHead(TtDeepLabV3PlusInsEmbedHead):
    """
    TTNN implementation of the Panoptic-DeepLab instance embedding head.
    This head predicts center heatmaps and offset vectors for instance segmentation.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        device: ttnn.Device,
        *,
        # --- Parameters for this class ---
        head_channels: int,
        norm: str,
        # --- Parameters that will be passed down to the base class ---
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        train_size: Optional[Tuple],
        # --- ALL weights for base class & this class ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        # --- Instance Embedding Head Specific Weights ---
        center_head_0_weight: torch.Tensor,
        center_head_1_weight: torch.Tensor,
        center_predictor_weight: torch.Tensor,
        offset_head_0_weight: torch.Tensor,
        offset_head_1_weight: torch.Tensor,
        offset_predictor_weight: torch.Tensor,
    ):
        # Call the base class __init__
        super().__init__(
            input_shape=input_shape,
            device=device,
            norm=norm,
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

        # Create the specific head and predictor layers for this class
        use_bias = norm == ""
        decoder_out_ch = decoder_channels[0]

        # Helper to create TTNN Conv2d
        def _create_tt_conv2d(
            weight: torch.Tensor, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, use_bias: bool
        ):
            param_dict = {"weight": weight}
            if use_bias:
                param_dict["bias"] = torch.zeros(1, 1, 1, out_ch)
            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
            return TtConv2d(parameters, stride=(stride, stride), padding=(padding, padding))

        # Center head layers
        self.center_head_0 = _create_tt_conv2d(center_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.center_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device)

        self.center_head_1 = _create_tt_conv2d(center_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.center_head_norm_1 = get_ttnn_norm(norm, head_channels, device)

        self.center_predictor = _create_tt_conv2d(center_predictor_weight, head_channels, 1, 1, 1, 0, True)

        # Offset head layers
        self.offset_head_0 = _create_tt_conv2d(offset_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.offset_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device)

        self.offset_head_1 = _create_tt_conv2d(offset_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.offset_head_norm_1 = get_ttnn_norm(norm, head_channels, device)

        self.offset_predictor = _create_tt_conv2d(offset_predictor_weight, head_channels, 2, 1, 1, 0, True)

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[Tuple[ttnn.Tensor, ttnn.Tensor], Dict]:
        """
        The forward pass for the Instance Embedding head in inference mode.
        """
        # 1. Get the final predictions from the full pipeline
        center_pred, offset_pred = self.layers(features)

        # 2. Perform the final upsampling to match the target output size
        center_pred = ttnn.to_layout(center_pred, ttnn.ROW_MAJOR_LAYOUT)
        center_pred = ttnn.upsample(center_pred, scale_factor=self.common_stride)
        center_pred = ttnn.to_layout(center_pred, ttnn.TILE_LAYOUT)

        offset_pred = ttnn.to_layout(offset_pred, ttnn.ROW_MAJOR_LAYOUT)
        offset_pred = ttnn.upsample(offset_pred, scale_factor=self.common_stride)
        offset_pred = ttnn.to_layout(offset_pred, ttnn.TILE_LAYOUT)

        # 3. Return the results in the ((center, offset), dict) format
        return (center_pred, offset_pred), {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        # 1. Get the refined feature map from the base class decoder
        y = super().layers(features)

        # 2. Apply the center head layers
        center_y = self.center_head_0(
            y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        center_y = self.center_head_norm_0(center_y)
        center_y = self.activation(center_y)

        center_y = self.center_head_1(
            center_y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        center_y = self.center_head_norm_1(center_y)
        center_y = self.activation(center_y)

        # 3. Apply the center predictor
        center_pred = self.center_predictor(center_y)

        # 4. Apply the offset head layers (starting from the same decoder output)
        offset_y = self.offset_head_0(
            y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        offset_y = self.offset_head_norm_0(offset_y)
        offset_y = self.activation(offset_y)

        offset_y = self.offset_head_1(
            offset_y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        offset_y = self.offset_head_norm_1(offset_y)
        offset_y = self.activation(offset_y)

        # 5. Apply the offset predictor
        offset_pred = self.offset_predictor(offset_y)

        # Clean up intermediate tensors
        ttnn.deallocate(center_y)
        ttnn.deallocate(offset_y)

        return center_pred, offset_pred
