# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers

from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.transformer import StateConversion, TransformerEncoder, TransformerEncoderConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


class Mistral3Encoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(hf_config: transformers.PretrainedConfig) -> TransformerEncoderConfig:
        """Takes a `Mistral3Config`; only its language model, `text_config`, is used."""
        text_config = hf_config.text_config
        if text_config.sliding_window is not None:
            msg = "sliding-window attention is not supported"
            raise ValueError(msg)

        rope_parameters = text_config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            msg = f"rope scaling is not supported, got {rope_parameters}"
            raise ValueError(msg)

        return TransformerEncoderConfig(
            vocab_size=text_config.vocab_size,
            head_size=text_config.head_dim,
            embed_size=text_config.hidden_size,
            ff_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            num_heads=text_config.num_attention_heads,
            num_kv_heads=text_config.num_key_value_heads,
            norm_eps=text_config.rms_norm_eps,
            attn_qkv_bias=False,
            attn_out_bias=False,
            rope_config=RopeConfig(theta=rope_parameters["rope_theta"]),
        )

    @staticmethod
    def convert_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return STATE_CONVERSION.convert(state_dict)


STATE_CONVERSION = StateConversion(
    rename=[
        (r"^model\.language_model\.embed_tokens", r"token_embedding"),
        (r"^model\.language_model\.layers\.([0-9]+)\.self_attn\.([qkvo])_proj", r"layers.\1.attn.\2_proj"),
        (r"^model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj", r"layers.\1.ff.gate"),
        (r"^model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj", r"layers.\1.ff.linear_in"),
        (r"^model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj", r"layers.\1.ff.linear_out"),
        (r"^model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm", r"layers.\1.ff_norm"),
        (r"^model\.language_model\.layers\.([0-9]+)\.input_layernorm", r"layers.\1.attn_norm"),
        (r"^model\.language_model\.norm\.weight", r"final_norm.weight"),
        (r"^lm_head\.weight", r"final_linear.weight"),
    ],
    remove=[
        r"^model\.vision_tower",
        r"^model\.multi_modal_projector",
    ],
)
