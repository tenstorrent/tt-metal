from typing import Dict, Optional, Tuple
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_aspp import get_ttnn_norm
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
        # NOVO: Svi torch.Tensor argumenti su zamijenjeni
        parameters,
        device: ttnn.Device,
        *,
        # Konfiguracioni parametri ostaju
        input_shape: Dict[str, ShapeSpec],
        head_channels: int,
        project_channels,
        aspp_dilations,
        aspp_dropout: float,
        decoder_channels,
        common_stride: int,
        norm: str,
        train_size: Optional[Tuple],
    ):
        super().__init__(
            parameters=parameters.decoder,
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
        )
        assert self.decoder_only
        use_bias = norm == ""
        decoder_out_ch = decoder_channels[0]
        logger.debug(f"Initializing TtPanopticDeepLabInsEmbedHead with head_channels: {head_channels}")

        # --- ISPRAVLJENO: Kreiranje slojeva sa ispravnim putanjama i provjerama ---

        # --- Center Prediction Grana ---
        # center_head_0
        center_head0_path = parameters.center_head[0]
        ch0_bias = center_head0_path.bias if "bias" in center_head0_path else None
        ch0_params = TtConv2dParameters(
            weight=center_head0_path.weight,
            bias=ch0_bias,
            device=self.device,
        )
        self.center_head_0 = TtConv2d.create_with_height_slicing(
            ch0_params, num_slices=2, stride=(1, 1), padding=(1, 1)
        )

        ch0_norm_params = center_head0_path.norm if "norm" in center_head0_path else None
        self.center_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=ch0_norm_params)

        # center_head_1
        center_head1_path = parameters.center_head[1]
        ch1_bias = center_head1_path.bias if "bias" in center_head1_path else None
        ch1_params = TtConv2dParameters(
            weight=center_head1_path.weight,
            bias=ch1_bias,
            device=self.device,
        )
        self.center_head_1 = TtConv2d.create_with_height_slicing(
            ch1_params, num_slices=2, stride=(1, 1), padding=(1, 1)
        )

        ch1_norm_params = center_head1_path.norm if "norm" in center_head1_path else None
        self.center_head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=ch1_norm_params)

        # center_predictor
        center_predictor_path = parameters.center_predictor
        cp_bias = center_predictor_path.bias if "bias" in center_predictor_path else None
        cp_params = TtConv2dParameters(
            weight=center_predictor_path.weight,
            bias=cp_bias,
            device=self.device,
        )
        self.center_predictor = TtConv2d.create(cp_params, stride=(1, 1), padding=(0, 0))

        # --- Offset Prediction Grana ---
        # offset_head_0
        offset_head0_path = parameters.offset_head[0]
        oh0_bias = offset_head0_path.bias if "bias" in offset_head0_path else None
        oh0_params = TtConv2dParameters(
            weight=offset_head0_path.weight,
            bias=oh0_bias,
            device=self.device,
        )
        self.offset_head_0 = TtConv2d.create_with_height_slicing(
            oh0_params, num_slices=2, stride=(1, 1), padding=(1, 1)
        )

        oh0_norm_params = offset_head0_path.norm if "norm" in offset_head0_path else None
        self.offset_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=oh0_norm_params)

        # offset_head_1
        offset_head1_path = parameters.offset_head[1]
        oh1_bias = offset_head1_path.bias if "bias" in offset_head1_path else None
        oh1_params = TtConv2dParameters(
            weight=offset_head1_path.weight,
            bias=oh1_bias,
            device=self.device,
        )
        self.offset_head_1 = TtConv2d.create_with_height_slicing(
            oh1_params, num_slices=2, stride=(1, 1), padding=(1, 1)
        )

        oh1_norm_params = offset_head1_path.norm if "norm" in offset_head1_path else None
        self.offset_head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=oh1_norm_params)

        # offset_predictor
        offset_predictor_path = parameters.offset_predictor
        op_bias = offset_predictor_path.bias if "bias" in offset_predictor_path else None
        op_params = TtConv2dParameters(
            weight=offset_predictor_path.weight,
            bias=op_bias,
            device=self.device,
        )
        self.offset_predictor = TtConv2d.create(op_params, stride=(1, 1), padding=(0, 0))

        # --- Inicijalizacija Upsample operacija ---
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

        # --- 2. Center Prediction Branch ---
        center_y = self.center_head_0(y)
        center_y = self.center_head_norm_0(center_y)
        center_y = self.activation(center_y)
        center_y = self.center_head_1(center_y)
        center_y = self.center_head_norm_1(center_y)
        center_y = self.activation(center_y)
        center_logits = self.center_predictor(center_y)

        # --- 3. Offset Prediction Branch ---
        offset_y = self.offset_head_0(y)
        offset_y = self.offset_head_norm_0(offset_y)
        offset_y = self.activation(offset_y)
        offset_y = self.offset_head_1(offset_y)
        offset_y = self.offset_head_norm_1(offset_y)
        offset_y = self.activation(offset_y)
        offset_logits = self.offset_predictor(offset_y)

        return center_logits, offset_logits
