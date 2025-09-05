import torch
from typing import Dict, List, Optional, Tuple
import ttnn

from .tt_aspp import get_ttnn_norm
from .tt_conv2dWrapper import TtConv2d, TtConv2dParameters
from .tt_semseg import TtDeepLabV3PlusHead
from .tt_pytorch_semSeg import ShapeSpec


class TtPanopticDeepLabInsEmbedHead(TtDeepLabV3PlusHead):
    """
    TTNN implementation for Panoptic-DeepLab instance embedding head.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        device: ttnn.Device,
        *,
        # --- Parametri specifični za ovu klasu ---
        head_channels: int,
        # --- Parametri koji se prosljeđuju baznoj klasi ---
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
        center_head_0_weight: torch.Tensor,
        center_head_1_weight: torch.Tensor,
        center_predictor_weight: torch.Tensor,
        offset_head_0_weight: torch.Tensor,
        offset_head_1_weight: torch.Tensor,
        offset_predictor_weight: torch.Tensor,
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

        def _create_tt_conv2d(
            weight: torch.Tensor, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, use_bias: bool
        ):
            param_dict = {"weight": weight}
            if use_bias:
                param_dict["bias"] = torch.zeros(1, 1, 1, out_ch)
            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
            return TtConv2d(parameters, stride=(stride, stride), padding=(padding, padding))

        # --- Center Prediction ---
        self.center_head_0 = _create_tt_conv2d(center_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.center_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=None)
        self.center_head_1 = _create_tt_conv2d(center_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.center_head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=None)
        self.center_predictor = _create_tt_conv2d(center_predictor_weight, head_channels, 1, 1, 1, 0, True)

        # --- Offset Prediction ---
        self.offset_head_0 = _create_tt_conv2d(offset_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.offset_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=None)
        self.offset_head_1 = _create_tt_conv2d(offset_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.offset_head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=None)
        self.offset_predictor = _create_tt_conv2d(offset_predictor_weight, head_channels, 2, 1, 1, 0, True)

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor, Dict, Dict]:
        center_logits, offset_logits = self.layers(features)

        # --- Final Upsample for Center ---
        center_logits = ttnn.to_layout(center_logits, ttnn.ROW_MAJOR_LAYOUT)
        center_logits = ttnn.upsample(center_logits, scale_factor=self.common_stride)
        center_logits = ttnn.to_layout(center_logits, ttnn.TILE_LAYOUT)

        # --- Final Upsample for Offset ---
        offset_logits = ttnn.to_layout(offset_logits, ttnn.ROW_MAJOR_LAYOUT)
        offset_logits = ttnn.upsample(offset_logits, scale_factor=self.common_stride)
        offset_logits = ttnn.mul(offset_logits, self.common_stride)
        offset_logits = ttnn.to_layout(offset_logits, ttnn.TILE_LAYOUT)

        return center_logits, offset_logits, {}, {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        y = super().layers(features)

        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

        # --- 2. Center Prediction Branch ---
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
        center_logits = self.center_predictor(center_y)

        # --- 3. Offset Prediction Branch ---
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
        offset_logits = self.offset_predictor(offset_y)

        return center_logits, offset_logits
