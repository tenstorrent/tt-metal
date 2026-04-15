# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight (CPU-only, no model download) regression tests for the
multimodal HF-key-remapping pipeline in load_checkpoints.py.
"""

from types import SimpleNamespace

import torch

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta_mllama,
    map_hf_to_meta_keys_mllama,
    split_hf_keys,
    standardize_hf_keys_multimodal,
)


def _make_mllama_config(num_hidden_layers=4, cross_attention_layers=None):
    """Build a minimal config object accepted by map_hf_to_meta_keys_mllama."""
    if cross_attention_layers is None:
        cross_attention_layers = [1]
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            cross_attention_layers=cross_attention_layers,
        )
    )


def _make_sample_mllama_state_dict():
    """Return a minimal HF-format state_dict that covers the projector keys
    plus the embed_tokens and lm_head keys required by map_hf_to_meta_keys_mllama."""
    t = torch.zeros(1)
    return {
        "model.multi_modal_projector.weight": t,
        "model.multi_modal_projector.bias": t,
        "model.vision_model.layernorm_pre.weight": t,
        "model.vision_model.layernorm_pre.bias": t,
        # Both must be present so standardize_hf_keys (called inside
        # standardize_hf_keys_multimodal) doesn't delete embed_tokens.
        "lm_head.weight": torch.zeros(16, 4),
        "model.embed_tokens.weight": torch.zeros(16, 4),
    }


class TestMllamaProjectorKeyRemap:
    """Ensure model.multi_modal_projector.* keys survive the two-stage
    multimodal pipeline and land as vision_model.vision_projection.*."""

    def test_projector_keys_after_full_pipeline(self):
        state_dict = _make_sample_mllama_state_dict()
        config = _make_mllama_config()

        state_dict = standardize_hf_keys_multimodal(state_dict)
        state_dict = split_hf_keys(state_dict)
        state_dict = map_hf_to_meta_keys_mllama(state_dict, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
        assert not any("multi_modal_projector" in k for k in state_dict)

    def test_projector_keys_via_convert_hf_to_meta_mllama(self):
        """Standardize_hf_keys_multimodal() -> convert_hf_to_meta_mllama().
        Asserts model.multi_modal_projector.weight ends up as
        vision_model.vision_projection.weight."""
        state_dict = _make_sample_mllama_state_dict()
        config = _make_mllama_config()
        head_dim = 64

        state_dict = standardize_hf_keys_multimodal(state_dict)
        state_dict = convert_hf_to_meta_mllama(state_dict, head_dim, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
        assert not any("multi_modal_projector" in k for k in state_dict)

    def test_projector_keys_without_standardize(self):
        """map_hf_to_meta_keys_mllama should also work when called directly
        with the original model.-prefixed keys (backward compat)."""
        t = torch.zeros(1)
        state_dict = {
            "model.multi_modal_projector.weight": t,
            "model.multi_modal_projector.bias": t,
            "model.embed_tokens.weight": torch.zeros(16, 4),
        }
        config = _make_mllama_config()

        state_dict = split_hf_keys(state_dict)
        state_dict = map_hf_to_meta_keys_mllama(state_dict, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
