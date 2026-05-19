# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# TT Mistral3 backbone: vision → projector → text LM.

from __future__ import annotations

import ttnn
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.pipeline.tt_ministral3_model import TtMinistral3Model
from models.experimental.devstarl2_small.tt.pipeline.tt_multimodal_projector import TTMistral3MultiModalProjector
from models.experimental.devstarl2_small.tt.pipeline.tt_pixtral_vision_model import TtPixtralVisionModel


class TtDevstral2SmallModel(LightweightModule):
    """Vision tower + multimodal projector + ``TtMinistral3Model`` (HF ``Mistral3Model``-style names)."""

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
    ):
        super().__init__()
        text_cfg = model_args.hf_config.text_config
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = dict(rope_params)

        self.mesh_device = mesh_device
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
        )
        if isinstance(text_cfg, Ministral3Config):
            lm_kwargs["ministral_text_config"] = text_cfg
        self.language_model = TtMinistral3Model(**lm_kwargs)

    def get_projected_image_features(self, pixel_values, image_sizes, position_ids_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Last vision hidden states through projector; matches HF ``get_image_features`` concatenation."""
        vt = self.vision_tower(pixel_values, image_sizes, position_ids_tt)
        seq_len, hidden = int(vt.shape[2]), int(vt.shape[3])
        tokens = ttnn.reshape(vt, (seq_len, hidden))
        return self.multi_modal_projector(tokens, image_sizes)


__all__ = ["TtDevstral2SmallModel"]
