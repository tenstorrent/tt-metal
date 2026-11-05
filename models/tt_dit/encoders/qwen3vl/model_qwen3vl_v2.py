# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel
from models.tt_dit.encoders.transformer import (
    WEIGHT_CACHE_DTYPE,
    StateConversion,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache


class Qwen3VlEncoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(hf_config: transformers.PretrainedConfig) -> TransformerEncoderConfig:
        """Takes a `Qwen3VLConfig`, or the `Qwen3VLTextConfig` under its `text_config`."""
        text_config = getattr(hf_config, "text_config", hf_config)
        rope = text_config.rope_parameters

        return TransformerEncoderConfig(
            vocab_size=text_config.vocab_size,
            head_size=text_config.head_dim,
            embed_size=text_config.hidden_size,
            ff_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            num_heads=text_config.num_attention_heads,
            num_kv_heads=text_config.num_key_value_heads,
            norm_eps=text_config.rms_norm_eps,
            attn_qkv_bias=text_config.attention_bias,
            attn_out_bias=False,
            # Qwen3 applies a per-head RMSNorm to the queries and keys.
            attn_qk_norm=True,
            rope_config=RopeConfig(
                theta=rope["rope_theta"],
                mrope_section=list(rope["mrope_section"]),
                mrope_interleaved=True,
            ),
        )

    @staticmethod
    def convert_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return STATE_CONVERSION.convert(state_dict)


# The keys of `Qwen3VLForConditionalGeneration`; the `model.` prefix is optional so a checkpoint
# of the bare `Qwen3VLModel` converts too.
STATE_CONVERSION = StateConversion(
    rename=[
        (r"^(?:model\.)?language_model\.embed_tokens", r"token_embedding"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.self_attn\.([qkvo])_proj", r"layers.\1.attn.\2_proj"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.self_attn\.([qk])_norm", r"layers.\1.attn.\2_norm"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.mlp\.gate_proj", r"layers.\1.ff.gate"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.mlp\.up_proj", r"layers.\1.ff.linear_in"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.mlp\.down_proj", r"layers.\1.ff.linear_out"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.post_attention_layernorm", r"layers.\1.ff_norm"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.input_layernorm", r"layers.\1.attn_norm"),
        (r"^(?:model\.)?language_model\.norm\.weight", r"final_norm.weight"),
        (r"^lm_head\.weight", r"final_linear.weight"),
    ],
    remove=[r"^(?:model\.)?visual"],
)


class Qwen3VlCheckpoint:
    """A Qwen3-VL checkpoint: fetches weights and builds a loaded ``Qwen3VlEncoder``.

    Reads only ``config.json`` in ``__init__``; the actual torch weights are loaded lazily, on
    ``build()`` cache-miss only. Only the language model is kept, so the vision tower is never
    instantiated.
    """

    def __init__(self, name: str, *, subfolder: str = "") -> None:
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder=subfolder)

        self._name = name
        self._subfolder = subfolder
        self.config = Qwen3VlEncoder.config_from_hf(hf_config)

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
    ) -> Qwen3VlEncoder:
        """Construct a ``Qwen3VlEncoder`` for this checkpoint and load its weights."""
        model = Qwen3VlEncoder(
            self.config,
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.load_weights(model, device=device, parallel_config=parallel_config)
        return model

    def load_weights(
        self,
        model: Qwen3VlEncoder,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        """Load the weights of a model built by ``build``, e.g. again after ``deallocate_weights``."""
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder=f"{self._subfolder}/text" if self._subfolder else "text",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
            dtype=WEIGHT_CACHE_DTYPE,
        )

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self._name,
            subfolder=self._subfolder,
            torch_dtype=torch.bfloat16,
        )
        return Qwen3VlEncoder.convert_state(torch_model.state_dict())


class Qwen3VlVisionCheckpoint:
    """The vision tower of a Qwen3-VL checkpoint: builds a loaded ``Qwen3VlVisionModel``.

    Reads only ``config.json`` in ``__init__``; the actual torch weights are loaded lazily, on
    ``build()`` cache-miss only.
    """

    # The tower's LayerNorms are built with this epsilon; the config does not carry it.
    NORM_EPS = 1e-6

    def __init__(self, name: str, *, subfolder: str = "") -> None:
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder=subfolder)

        self._name = name
        self._subfolder = subfolder
        self.config = hf_config.vision_config

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
    ) -> Qwen3VlVisionModel:
        """Construct a ``Qwen3VlVisionModel`` for this checkpoint and load its weights.

        The tower is tensor parallel as ``parallel_config`` says, and sequence parallel if it says so.
        """
        config = self.config
        model = Qwen3VlVisionModel(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            depth=config.depth,
            intermediate_size=config.intermediate_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            num_position_embeddings=config.num_position_embeddings,
            out_hidden_size=config.out_hidden_size,
            hidden_act=config.hidden_act,
            norm_eps=self.NORM_EPS,
            deepstack_visual_indexes=config.deepstack_visual_indexes,
            mesh_device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder=f"{self._subfolder}/vision" if self._subfolder else "vision",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
        )
        return model

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self._name,
            subfolder=self._subfolder,
            torch_dtype=torch.bfloat16,
        )
        return dict(torch_model.model.visual.state_dict())
