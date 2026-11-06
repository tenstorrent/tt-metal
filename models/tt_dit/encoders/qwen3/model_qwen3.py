# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
import transformers

from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.transformer import StateConversion, TransformerEncoder, TransformerEncoderConfig


class Qwen3Encoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(hf_config: transformers.PretrainedConfig) -> TransformerEncoderConfig:
        if hf_config.use_sliding_window:
            msg = "sliding-window attention is not supported"
            raise ValueError(msg)

        rope_parameters = hf_config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            msg = f"rope scaling is not supported, got {rope_parameters}"
            raise ValueError(msg)

        return TransformerEncoderConfig(
            vocab_size=hf_config.vocab_size,
            head_size=hf_config.head_dim,
            embed_size=hf_config.hidden_size,
            ff_size=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            norm_eps=hf_config.rms_norm_eps,
            attn_qkv_bias=hf_config.attention_bias,
            attn_out_bias=hf_config.attention_bias,
            attn_qk_norm=True,
            rope_config=RopeConfig(theta=rope_parameters["rope_theta"]),
            # The text encoders only need hidden states; the head is tied to the embedding and
            # has no weight of its own in the checkpoint.
            final_linear=False,
        )

    @staticmethod
    def convert_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return STATE_CONVERSION.convert(state_dict)


STATE_CONVERSION = StateConversion(
    rename=[
        (r"^model\.embed_tokens", r"token_embedding"),
        (r"^model\.layers\.([0-9]+)\.self_attn\.([qkvo])_proj", r"layers.\1.attn.\2_proj"),
        (r"^model\.layers\.([0-9]+)\.self_attn\.([qk])_norm", r"layers.\1.attn.\2_norm"),
        (r"^model\.layers\.([0-9]+)\.mlp\.gate_proj", r"layers.\1.ff.gate"),
        (r"^model\.layers\.([0-9]+)\.mlp\.up_proj", r"layers.\1.ff.linear_in"),
        (r"^model\.layers\.([0-9]+)\.mlp\.down_proj", r"layers.\1.ff.linear_out"),
        (r"^model\.layers\.([0-9]+)\.post_attention_layernorm", r"layers.\1.ff_norm"),
        (r"^model\.layers\.([0-9]+)\.input_layernorm", r"layers.\1.attn_norm"),
        (r"^model\.norm\.weight", r"final_norm.weight"),
    ],
    remove=[
        r"^lm_head\.",
    ],
)
