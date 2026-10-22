# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict


def build_vision_encoder(args, dtype, state_dict: Dict[str, Any], weight_cache_path: Path):
    """
    Factory function to create vision encoders

    Design pattern: Factory pattern for modular encoder selection

    Args:
        args: VisionModelArgs with encoder configuration
        dtype: ttnn data type (ttnn.bfloat16)
        state_dict: Model weights dictionary
        weight_cache_path: Path to cached weights

    Returns:
        BaseVisionEncoder: Concrete encoder instance

    Supported encoders:
        - "clip": CLIP ViT-L/14 @ 336px (LLaVA 1.5 default)
        - "siglip": SigLIP encoder (LLaVA-MORE)
        - Future: "dino", "ijepa", "vit", "onevision"

    Usage:
        encoder = build_vision_encoder(args, dtype, state_dict, cache_path)
        vision_tokens = encoder(images)

    Reference: LLaVA-MORE/src/llava/model/multimodal_encoder/builder.py
    """
    encoder_type = args.vision_encoder_type.lower()

    if encoder_type == "clip":
        from .clip_encoder import CLIPVisionEncoder

        return CLIPVisionEncoder(args, state_dict, weight_cache_path)

    elif encoder_type == "siglip":
        from .siglip_encoder import SigLIPVisionEncoder

        return SigLIPVisionEncoder(args, state_dict, weight_cache_path)

    # Future encoders
    # elif encoder_type == "dino":
    #     from .dino_encoder import DINOVisionEncoder
    #     return DINOVisionEncoder(args, dtype, state_dict, weight_cache_path)
    #
    # elif encoder_type == "ijepa":
    #     from .ijepa_encoder import IJepaVisionEncoder
    #     return IJepaVisionEncoder(args, dtype, state_dict, weight_cache_path)

    else:
        raise ValueError(
            f"Unknown vision encoder type: {encoder_type}. "
            f"Supported: clip, siglip. "
            f"Future: dino, ijepa, vit, onevision"
        )
