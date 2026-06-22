# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# TT Mistral3 backbone: vision → projector → text LM + LM head.

from __future__ import annotations

import ttnn
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_small.devstral_utils.multimodal_demo_helpers import (
    demo_lm_head_max_columns_per_device,
    resolve_rope_parameters,
)
from models.experimental.devstral2_small.tt.pipeline.tt_ministral3_model import TtMinistral3Model
from models.experimental.devstral2_small.tt.pipeline.tt_multimodal_projector import TTMistral3MultiModalProjector
from models.experimental.devstral2_small.tt.pipeline.tt_pixtral_vision_model import TtPixtralVisionModel
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead


class TtDevstral2SmallModel(LightweightModule):
    """Vision tower + multimodal projector + text decode logits."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        weight_cache_path,
        dtype,
        transformation_mats,
        configuration,
        vision_config,
        vision_n_layers: int | None = None,
        embed_dtype=None,
        paged_attention_config=None,
        lm_head_dtype=None,
        lm_head_max_columns_per_device: int | None = None,
    ):
        super().__init__()
        text_cfg = model_args.hf_config.text_config
        rope_params = resolve_rope_parameters(text_cfg)

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.model_args = model_args
        self.vision_tower = TtPixtralVisionModel(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=meta_state_dict,
            configuration=model_args,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            vision_config=vision_config,
            n_layers=vision_n_layers,
        )
        self.multi_modal_projector = TTMistral3MultiModalProjector(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=meta_state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            eps=float(text_cfg.rms_norm_eps),
        )
        lm_kwargs = dict(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            model_args=model_args,
            meta_state_dict=meta_state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=configuration,
            llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
            original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
            embed_dtype=embed_dtype,
            paged_attention_config=paged_attention_config,
        )
        if isinstance(text_cfg, Ministral3Config):
            lm_kwargs["ministral_text_config"] = text_cfg
        self._lm_kwargs = lm_kwargs
        self._language_model: TtMinistral3Model | None = None
        lm_head_dtype = dtype if lm_head_dtype is None else lm_head_dtype
        sd_prefix = model_args.get_state_dict_prefix("", None)
        self._lm_head_kwargs = dict(
            args=model_args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=lm_head_dtype,
            state_dict=meta_state_dict,
            state_dict_prefix=sd_prefix,
            weight_cache_path=weight_cache_path,
            max_columns_per_device=(
                demo_lm_head_max_columns_per_device(model_args)
                if lm_head_max_columns_per_device is None
                else lm_head_max_columns_per_device
            ),
        )
        self._lm_head: LMHead | None = None

    @property
    def language_model(self) -> TtMinistral3Model:
        if self._language_model is None:
            self._language_model = TtMinistral3Model(**self._lm_kwargs)
        return self._language_model

    @language_model.setter
    def language_model(self, value: TtMinistral3Model) -> None:
        self._language_model = value

    @property
    def lm_head(self) -> LMHead:
        if self._lm_head is None:
            out_key = f"{self._lm_head_kwargs['state_dict_prefix']}output.weight"
            if out_key not in self._lm_head_kwargs["state_dict"]:
                raise RuntimeError(f"Missing {out_key!r} in meta state dict (required for LM head).")
            self._lm_head = LMHead(**self._lm_head_kwargs)
        return self._lm_head

    @lm_head.setter
    def lm_head(self, value: LMHead) -> None:
        self._lm_head = value

    def lm_head_forward(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Run TT LM head on normalized text hidden states and return DRAM logits."""
        h = hidden
        lm_head_input_mem_cfg = self.model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
        if lm_head_input_mem_cfg is not None and lm_head_input_mem_cfg.is_sharded():
            h = ttnn.interleaved_to_sharded(hidden, lm_head_input_mem_cfg)
            ttnn.deallocate(hidden)
        logits = self.lm_head(h)
        return ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward_decode_from_device_tensors(
        self,
        token_ids_tt: ttnn.Tensor,
        pos_uint32: ttnn.Tensor,
        pos_int32: ttnn.Tensor,
        page_table=None,
    ) -> ttnn.Tensor:
        hidden = self.language_model.forward_decode_from_device_tensors(
            token_ids_tt, pos_uint32, pos_int32, page_table=page_table
        )
        return self.lm_head_forward(hidden)

    def forward_decode(self, token_ids_tt: ttnn.Tensor, decode_pos: int) -> ttnn.Tensor:
        hidden = self.language_model.forward_decode(token_ids_tt, decode_pos)
        return self.lm_head_forward(hidden)

    def get_projected_image_features(self, pixel_values, image_sizes, position_ids) -> ttnn.Tensor:
        """Last vision hidden states through projector; matches HF ``get_image_features`` concatenation."""
        vt = self.vision_tower(pixel_values, image_sizes, position_ids)
        seq_len, hidden = int(vt.shape[2]), int(vt.shape[3])
        tokens = ttnn.reshape(vt, (seq_len, hidden))
        if tokens.memory_config().buffer_type == ttnn.BufferType.L1:
            tokens = ttnn.to_memory_config(tokens, ttnn.DRAM_MEMORY_CONFIG)
        return self.multi_modal_projector(tokens, image_sizes)


__all__ = ["TtDevstral2SmallModel"]
