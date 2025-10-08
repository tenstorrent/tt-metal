# SPDX-FileCopyrightText: Copyright (c) 2025 Motif Technologies

# SPDX-License-Identifier: MIT License

"""Configuration dataclass for Motif Image (MMDiT + encoders/decoder).

This module provides a lightweight, JSON-serializable configuration used to construct the
inference model and control backbone options such as attention mode, register tokens, and
positional embeddings.
"""

import json
from dataclasses import dataclass

ENCODED_TEXT_DIM = 4096
POOLED_TEXT_DIM = 2048
VAE_COMPRESSION_RATIO = 8


@dataclass
class MotifImageConfig:
    """Configuration for `MotifImage` and the underlying `MotifDiT`.

    Key groups:
    - General/model: transformer width/depth, patch size, spatial resolution, VAE options
    - Blocks: dims for time/text embeddings and MLP, attention backend/mode
    - Attention: head counts and dropout
    - Modulation: FiLM/ADAiN/SPADE selector
    - Register tokens: counts for global tokens used during high-res training
    - Feature alignment: optional DINOv2-based guidance during training (unused at inference)
    - Personalization/PEFT: LoRA-related defaults for preference training

    Instances can be loaded via `from_json_file` to keep configs portable in repos and CLIs.
    """

    # General
    num_layers: int = 12
    hidden_dim: int = 768  # common hidden dimension for the transformer arch
    patch_size: int = 2
    image_dim: int = 224
    in_channel: int = 4
    out_channel: int = 4
    modulation_dim: int = ENCODED_TEXT_DIM  # input dimension for modulation layer (shifting and scaling)
    height: int = 1024
    width: int = 1024
    vae_compression: int = VAE_COMPRESSION_RATIO  # reducing resolution with the VAE
    vae_type: str = "SD3"  # SDXL or SD3
    pos_emb_size: int = None
    conv_header: bool = False

    # Outside of the MMDiT block
    time_embed_dim: int = 2048  # Initial projection (discrete_time embedding) output dim
    pooled_text_dim: int = POOLED_TEXT_DIM
    text_emb_dim: int = 768

    # MMDiTBlock
    t_emb_dim: int = 256
    attn_embed_dim: int = 768  # hidden dimension during the attention
    mlp_hidden_dim: int = 2048
    attn_mode: str = None  # {'flash', 'sdpa', None}
    use_final_layer_norm: bool = False
    use_time_token_in_attn: bool = False

    # GroupedQueryAttention
    num_attention_heads: int = 12
    num_key_value_heads: int = 6
    use_scaled_dot_product_attention: bool = True
    dropout: float = 0.0

    # Modulation
    use_modulation: bool = True
    modulation_type: str = "film"  # Choose from 'film', 'adain', or 'spade'

    # Register tokens
    register_token_num: int = 4
    additional_register_token_num: int = 12

    # use dinov2 feature-align loss
    dinov2_feature_align_loss: bool = False
    feature_align_loss_weight: float = 0.5
    num_feature_align_layers: int = 8  # number of transformer layers to calculate feature-align loss

    # Personalization related
    image_encoder_name: str = None  # if set, the persoanlized image encoder will be loaded
    freeze_dit_backbone: bool = False

    # Preference optimization
    preference_train: bool = False
    lora_rank: int = 64
    lora_alpha: int = 8

    skip_register_token_num: int = 0

    @classmethod
    def from_json_file(cls, json_file):
        """
        Instantiates a configuration from a JSON file path.

        Args:
            json_file: Path to the JSON file containing parameters.

        Returns:
            MotifImageConfig: Parsed configuration instance.
        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
