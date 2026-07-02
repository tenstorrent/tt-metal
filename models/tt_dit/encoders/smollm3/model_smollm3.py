# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.transformer import StateConversion, TransformerEncoder, TransformerEncoderConfig
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache


class SmolLm3Encoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(hf_config: transformers.PretrainedConfig) -> TransformerEncoderConfig:
        if hf_config.use_sliding_window:
            msg = "sliding-window attention is not supported"
            raise ValueError(msg)

        if not all(t == "full_attention" for t in hf_config.layer_types):
            msg = f"expected all layer_types to be 'full_attention', got {set(hf_config.layer_types)}"
            raise ValueError(msg)

        return TransformerEncoderConfig(
            vocab_size=hf_config.vocab_size,
            head_size=hf_config.hidden_size // hf_config.num_attention_heads,
            embed_size=hf_config.hidden_size,
            ff_size=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            norm_eps=hf_config.rms_norm_eps,
            attn_qkv_bias=False,
            attn_out_bias=False,
            rope_config=RopeConfig(theta=hf_config.rope_parameters["rope_theta"]),
            nope_layer_indices=[i for i, uses_rope in enumerate(hf_config.no_rope_layers) if not uses_rope],
        )

    @staticmethod
    def convert_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return STATE_CONVERSION.convert(state_dict)


STATE_CONVERSION = StateConversion(
    rename=[
        (r"^model\.embed_tokens", r"token_embedding"),
        (r"^model\.layers\.([0-9]+)\.self_attn\.([qkvo])_proj", r"layers.\1.attn.\2_proj"),
        (r"^model\.layers\.([0-9]+)\.mlp\.gate_proj", r"layers.\1.ff.gate"),
        (r"^model\.layers\.([0-9]+)\.mlp\.up_proj", r"layers.\1.ff.linear_in"),
        (r"^model\.layers\.([0-9]+)\.mlp\.down_proj", r"layers.\1.ff.linear_out"),
        (r"^model\.layers\.([0-9]+)\.post_attention_layernorm", r"layers.\1.ff_norm"),
        (r"^model\.layers\.([0-9]+)\.input_layernorm", r"layers.\1.attn_norm"),
        (r"^model\.norm\.weight", r"final_norm.weight"),
        (r"^lm_head\.weight", r"final_linear.weight"),
    ],
)


class SmolLm3Checkpoint:
    """A SmolLM3 text-encoder checkpoint: fetches weights and builds a loaded ``SmolLm3Encoder``.

    Reads only ``config.json`` in ``__init__``; the actual torch weights are loaded lazily, on
    ``build()`` cache-miss only.
    """

    def __init__(self, name: str) -> None:
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder="text_encoder")

        self._name = name
        self.config = SmolLm3Encoder.config_from_hf(hf_config)

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> SmolLm3Encoder:
        """Construct a ``SmolLm3Encoder`` for this checkpoint and load its weights."""
        model = SmolLm3Encoder(
            self.config,
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder="text_encoder",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
        )
        return model

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        return SmolLm3Encoder.convert_state(torch_model.state_dict())
