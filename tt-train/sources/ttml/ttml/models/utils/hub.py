# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HF models by name: a hub id, or a local directory laid out as ``save_pretrained`` writes one."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

# What a model is built from: the weights, the configs and the tokenizer.
_MODEL_FILES = ["*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"]


def load_hf_config(model_name: str) -> dict[str, Any]:
    """The ``config.json`` of a local model directory, or of a hub model (downloads only that file)."""
    path = Path(model_name) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model_name, "config.json"))
    return json.loads(path.read_text())


def model_dir(model_name: str, local_only: bool = False) -> str:
    """A local directory with the model's weights, configs and tokenizer; downloads a hub model once.

    With ``local_only``, a hub model is never looked up online: the copy downloaded before is taken.
    """
    if Path(model_name).is_dir():
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, allow_patterns=_MODEL_FILES, local_files_only=local_only)


def model_revision(model_name: str) -> Optional[str]:
    """The hub commit of the copy of a hub model that ``model_dir`` downloaded; None for a local directory."""
    if Path(model_name).is_dir():
        return None
    # The hub cache keeps each commit in snapshots/<commit>/.
    return Path(model_dir(model_name, local_only=True)).name


def same_model(base: Any, model_name: str) -> bool:
    """Whether a checkpoint's base model ``base`` names ``model_name``: a hub id and a local copy of it match, and so
    does a checkpoint that records no base."""
    if not base:
        return True

    def tail(name: str) -> str:
        return os.path.basename(os.path.normpath(name))

    return base == model_name or tail(base) == tail(model_name)
