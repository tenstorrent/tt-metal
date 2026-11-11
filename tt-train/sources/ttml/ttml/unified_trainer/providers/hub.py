# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HF models by hub id or local directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# What a training run reads from a hub model: the weights, the configs and the tokenizer.
_MODEL_FILES = ["*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"]


def load_hf_config(model_name: str) -> dict[str, Any]:
    """The ``config.json`` of a local model directory, or of a hub model (downloads only that file)."""
    path = Path(model_name) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model_name, "config.json"))
    return json.loads(path.read_text())


def model_dir(model_name: str) -> str:
    """A local directory with the model's weights, configs and tokenizer; downloads a hub model once."""
    if Path(model_name).is_dir():
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, allow_patterns=_MODEL_FILES)
