import torch
from typing import Dict, List, Optional, Tuple
import ttnn

# Importujte sve potrebne klase i funkcije iz vašeg projekta
from .tt_aspp import get_ttnn_norm
from .tt_conv2dWrapper import TtConv2d, TtConv2dParameters
from .tt_semseg import TtDeepLabV3PlusHead  # Importujemo bazu
from .tt_pytorch_semSeg import ShapeSpec


class TtPanopticDeepLabInsEmbedHead(TtDeepLabV3PlusHead):
    """
    TTNN implementacija Panoptic-DeepLab instance embedding glave.
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
        # --- SVE težine za baznu klasu i ovu klasu ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        # --- SVE težine specifične za ovu instance head ---
        center_head_0_weight: torch.Tensor,
        center_head_1_weight: torch.Tensor,
        center_predictor_weight: torch.Tensor,
        offset_head_0_weight: torch.Tensor,
        offset_head_1_weight: torch.Tensor,
        offset_predictor_weight: torch.Tensor,
    ):
        # Pozivamo __init__ bazne klase da napravi zajednički dekoder
        super().__init__(
            input_shape=input_shape,
            device=device,
            norm=norm,
            num_classes=None,  # Ključno za decoder_only mod
            predictor_weight=None,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            train_size=train_size,
            # Prosljeđujemo sve težine za dekoder
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            project_conv_weights=project_conv_weights,
            fuse_conv_0_weights=fuse_conv_0_weights,
            fuse_conv_1_weights=fuse_conv_1_weights,
        )
        assert self.decoder_only

        # Pomoćna funkcija za kreiranje TtConv2d slojeva
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

        # --- Kreiranje Center Prediction grane ---
        self.center_head_0 = _create_tt_conv2d(center_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.center_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=None)
        self.center_head_1 = _create_tt_conv2d(center_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.center_head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=None)
        self.center_predictor = _create_tt_conv2d(center_predictor_weight, head_channels, 1, 1, 1, 0, True)

        # --- Kreiranje Offset Prediction grane ---
        self.offset_head_0 = _create_tt_conv2d(offset_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.offset_head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device, norm_params=None)
        self.offset_head_1 = _create_tt_conv2d(offset_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.offset_head_norm_1 = get_ttnn_norm(norm, head_channels, device, norm_params=None)
        self.offset_predictor = _create_tt_conv2d(offset_predictor_weight, head_channels, 2, 1, 1, 0, True)

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor, Dict, Dict]:
        """
        Forward pass za Instance Embedding glavu u inference modu.
        Vraća upsample-ovane center i offset predikcije.
        """
        center_logits, offset_logits = self.layers(features)

        # --- Finalni Upsample za Center ---
        center_logits = ttnn.to_layout(center_logits, ttnn.ROW_MAJOR_LAYOUT)
        # Ovdje možete dodati slicing logiku ako bude potrebe (kao u SemSeg)
        # center_logits = ttnn.pad(center_logits, [[0,0],[0,0],[0,0],[0, ], value=0)  # Padding do 24 kanala
        # y = ttnn.pad(y, padding=[[0,0],[0,0],[0,0],[0, 24 - original_channels]], value=0)
        center_logits = ttnn.upsample(center_logits, scale_factor=self.common_stride)
        center_logits = ttnn.to_layout(center_logits, ttnn.TILE_LAYOUT)

        # --- Finalni Upsample za Offset ---
        offset_logits = ttnn.to_layout(offset_logits, ttnn.ROW_MAJOR_LAYOUT)
        # Ovdje možete dodati slicing logiku ako bude potrebe
        offset_logits = ttnn.upsample(offset_logits, scale_factor=self.common_stride)
        # Za offset se rezultat množi sa stride-om
        offset_logits = ttnn.mul(offset_logits, self.common_stride)
        offset_logits = ttnn.to_layout(offset_logits, ttnn.TILE_LAYOUT)

        return center_logits, offset_logits, {}, {}

    def layers(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Pokreće zajednički dekoder, a zatim paralelne instance embedding grane.
        """
        # 1. Dobijamo izlaz iz zajedničkog dekodera pozivom bazne klase
        y = super().layers(features)

        # Opcionalno, prebaciti y u DRAM da bude dostupan objema granama
        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

        # --- 2. Center Prediction Grana ---
        # Koristimo slice_config kao u SemSeg, jer je ulaz isti
        print(y)
        center_y = self.center_head_0(
            y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        center_y = self.center_head_norm_0(center_y)
        center_y = self.activation(center_y)
        print(center_y)
        center_y = self.center_head_1(
            center_y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        center_y = self.center_head_norm_1(center_y)
        center_y = self.activation(center_y)
        print(center_y)
        center_logits = self.center_predictor(center_y)

        # --- 3. Offset Prediction Grana ---
        # Koristi isti ulaz 'y'
        print("KRECE PREDECTION")
        print(y)
        offset_y = self.offset_head_0(
            y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        offset_y = self.offset_head_norm_0(offset_y)
        offset_y = self.activation(offset_y)
        print(offset_y)
        offset_y = self.offset_head_1(
            offset_y,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            ),
        )
        offset_y = self.offset_head_norm_1(offset_y)
        offset_y = self.activation(offset_y)
        print(offset_y)
        offset_logits = self.offset_predictor(offset_y)

        return center_logits, offset_logits
