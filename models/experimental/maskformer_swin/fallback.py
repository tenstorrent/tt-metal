# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fallback MaskFormer pipeline built on HuggingFace modules.

This helper orchestrates the Swin backbone, pixel decoder, transformer decoder,
and heads implemented in this package, using the HuggingFace reference weights.
It enables CPU-only validation before TT-NN kernels are in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import copy

import torch

try:
    from transformers import AutoImageProcessor, MaskFormerConfig
    from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput
except ModuleNotFoundError:  # pragma: no cover - soft dependency for fallback.
    AutoImageProcessor = None
    MaskFormerConfig = None
    MaskFormerForInstanceSegmentationOutput = None

from .backbone_swin import MaskFormerSwinBackbone, SwinBackboneConfig
from .pixel_decoder import MaskFormerPixelDecoder, PixelDecoderConfig
from .transformer_decoder import MaskFormerTransformerDecoder, TransformerDecoderConfig
from .heads import MaskFormerHeads, MaskFormerHeadsConfig
from .weights import ReferenceWeights
from .ttnn_compat import ttnn


@dataclass
class FallbackOutputs:
    class_logits: torch.Tensor
    mask_logits: torch.Tensor
    encoder_feature_maps: List[torch.Tensor]
    mask_features: torch.Tensor
    pixel_decoder_hidden_states: List[torch.Tensor]
    transformer_hidden_states: List[torch.Tensor]


class MaskFormerFallbackPipeline:
    """CPU-based MaskFormer inference composed of the fallback modules."""

    def __init__(
        self,
        *,
        backbone: MaskFormerSwinBackbone,
        pixel_decoder: MaskFormerPixelDecoder,
        transformer_decoder: MaskFormerTransformerDecoder,
        heads: MaskFormerHeads,
        config: MaskFormerConfig,
    ) -> None:
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.heads = heads
        self.config = config

    @classmethod
    def from_reference(
        cls,
        reference: ReferenceWeights,
        state_dict: Dict[str, object],
        *,
        device: Optional[torch.device] = None,
    ) -> "MaskFormerFallbackPipeline":
        if MaskFormerConfig is None:
            raise RuntimeError("transformers is required to build the fallback pipeline.")

        config_payload = dict(reference.config)
        backbone_value = reference.config.get("backbone_config", {})
        if isinstance(backbone_value, dict):
            backbone_cfg = dict(backbone_value)
        else:
            backbone_cfg = backbone_value.to_dict() if hasattr(backbone_value, "to_dict") else {}
        backbone_cfg.setdefault("model_type", "swin")
        config_payload["backbone_config"] = backbone_cfg
        decoder_value = reference.config.get("decoder_config", {})
        if isinstance(decoder_value, dict):
            decoder_cfg = dict(decoder_value)
        else:
            decoder_cfg = decoder_value.to_dict() if hasattr(decoder_value, "to_dict") else {}
        decoder_cfg.setdefault("model_type", "detr")
        config_payload["decoder_config"] = decoder_cfg

        sw_cfg = SwinBackboneConfig.from_hf_dict(backbone_cfg)
        backbone = MaskFormerSwinBackbone.from_huggingface(state_dict, device=device, config_dict=backbone_cfg)

        pixel_cfg = PixelDecoderConfig(fpn_dim=config_payload.get("fpn_feature_size", 256))
        pixel_decoder = MaskFormerPixelDecoder.from_huggingface(state_dict, config=pixel_cfg, device=device)

        transformer_cfg = TransformerDecoderConfig(
            num_layers=config_payload.get("num_hidden_layers", 6),
            num_attention_heads=config_payload.get("num_attention_heads", 8),
            hidden_dim=config_payload.get("fpn_feature_size", 256),
            dim_feedforward=decoder_cfg.get("decoder_ffn_dim", 2048),
            dropout=decoder_cfg.get("dropout", 0.0),
            activation=decoder_cfg.get("activation_function", "relu"),
            in_features=pixel_cfg.input_channels[-1],
            maskformer_config=copy.deepcopy(config_payload),
        )
        transformer_decoder = MaskFormerTransformerDecoder.from_huggingface(
            state_dict, config=transformer_cfg, device=device
        )

        heads_cfg = MaskFormerHeadsConfig(
            num_classes=len(config_payload.get("id2label", {})),
            hidden_dim=transformer_cfg.hidden_dim,
            mask_dim=pixel_cfg.fpn_dim,
        )
        heads = MaskFormerHeads(config=heads_cfg, device=device)
        heads.load_weights(state_dict)

        hf_model_config = MaskFormerConfig(**copy.deepcopy(config_payload))
        return cls(
            backbone=backbone,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            heads=heads,
            config=hf_model_config,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> FallbackOutputs:
        features, encoder_hidden = self.backbone.forward(pixel_values)
        mask_features, pixel_decoder_hidden = self.pixel_decoder.forward(features)
        use_tt_decoder = os.environ.get("MASKFORMER_TT_DECODER") == "1"
        if use_tt_decoder and ttnn is not None:
            transformer_last, transformer_hidden, attentions = self.transformer_decoder.forward_tt(
                features[-1],
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        else:
            transformer_last, transformer_hidden, attentions = self.transformer_decoder.forward(
                features[-1],
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        class_logits, mask_logits = self.heads.forward(transformer_last, mask_features)
        _ = attentions  # currently unused; kept for parity
        return FallbackOutputs(
            class_logits=class_logits,
            mask_logits=mask_logits,
            encoder_feature_maps=features,
            mask_features=mask_features,
            pixel_decoder_hidden_states=pixel_decoder_hidden,
            transformer_hidden_states=transformer_hidden,
        )

    def post_process_semantic(
        self,
        outputs: FallbackOutputs,
        *,
        image_processor: AutoImageProcessor,
        target_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        if MaskFormerForInstanceSegmentationOutput is None:
            raise RuntimeError("transformers is required for post-processing.")

        mf_output = MaskFormerForInstanceSegmentationOutput(
            class_queries_logits=outputs.class_logits,
            masks_queries_logits=outputs.mask_logits,
        )
        return image_processor.post_process_semantic_segmentation(mf_output, target_sizes=target_sizes)

    def post_process_panoptic(
        self,
        outputs: FallbackOutputs,
        *,
        image_processor: AutoImageProcessor,
        target_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        if MaskFormerForInstanceSegmentationOutput is None:
            raise RuntimeError("transformers is required for post-processing.")

        mf_output = MaskFormerForInstanceSegmentationOutput(
            class_queries_logits=outputs.class_logits,
            masks_queries_logits=outputs.mask_logits,
        )
        # Each item: {segmentation: Tensor[H,W], segments_info: list[{id, category_id, ...}]}
        return image_processor.post_process_panoptic_segmentation(mf_output, target_sizes=target_sizes)
