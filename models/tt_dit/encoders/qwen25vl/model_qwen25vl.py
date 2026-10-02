# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.transformer import (
    WEIGHT_CACHE_DTYPE,
    StateConversion,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache


class Qwen25VlEncoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(hf_config: transformers.PretrainedConfig) -> TransformerEncoderConfig:
        """Takes the whole `Qwen2_5_VLConfig`; the language-model fields live under `text_config`."""
        text_config = hf_config.text_config

        # The checkpoint's multimodal rope splits the head across three position axes. Without
        # images all three carry the same sequence positions, so it reduces to plain rope.
        rope_theta = text_config.rope_parameters["rope_theta"]

        return TransformerEncoderConfig(
            vocab_size=text_config.vocab_size,
            head_size=text_config.hidden_size // text_config.num_attention_heads,
            embed_size=text_config.hidden_size,
            ff_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            num_heads=text_config.num_attention_heads,
            num_kv_heads=text_config.num_key_value_heads,
            norm_eps=text_config.rms_norm_eps,
            # Qwen2.5 carries a QKV bias, which Qwen3 drops.
            attn_qkv_bias=True,
            attn_out_bias=False,
            # At bfloat8 the QKV projection alone costs the prompt embeddings two points of PCC.
            attn_qkv_dtype=ttnn.bfloat16,
            rope_config=RopeConfig(theta=rope_theta),
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
    remove=[r"^model\.visual"],
)


class Qwen25VlCheckpoint:
    """A Qwen2.5-VL checkpoint: fetches weights and builds a loaded ``Qwen25VlEncoder``.

    Reads only ``config.json`` in ``__init__``; the actual torch weights are loaded lazily, on
    ``build()`` cache-miss only. Only the language model is kept, so the vision tower is never
    instantiated.
    """

    def __init__(self, name: str, *, subfolder: str = "") -> None:
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder=subfolder)

        self._name = name
        self._subfolder = subfolder
        self.config = Qwen25VlEncoder.config_from_hf(hf_config)

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
    ) -> Qwen25VlEncoder:
        """Construct a ``Qwen25VlEncoder`` for this checkpoint and load its weights."""
        model = Qwen25VlEncoder(
            self.config,
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.load_weights(model, device=device, parallel_config=parallel_config)
        return model

    def load_weights(
        self,
        model: Qwen25VlEncoder,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        """Load the weights of a model built by ``build``, e.g. again after ``deallocate_weights``."""
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder=self._subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
            dtype=f"{WEIGHT_CACHE_DTYPE}_qkv16",
        )

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._name,
            subfolder=self._subfolder,
            torch_dtype=torch.bfloat16,
        )
        return Qwen25VlEncoder.convert_state(torch_model.state_dict())
