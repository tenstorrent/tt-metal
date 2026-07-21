# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify vendored VibeVoice-1.5B reference code loads weights from MODEL_PATH."""

from pathlib import Path

import torch

from models.experimental.vibevoice.common.config import MODEL_PATH
from modular.modeling_vibevoice import VibeVoiceForConditionalGeneration


def test_model_load_has_qwen_backbone():
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )

    assert hasattr(model, "model")
    lm = model.model.language_model
    assert lm.config.num_hidden_layers == 28
    assert lm.config.hidden_size == 1536


def test_config_json_present():
    assert (Path(MODEL_PATH) / "config.json").is_file()
