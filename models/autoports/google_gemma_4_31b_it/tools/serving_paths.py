#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resolve the pinned local checkpoint and the actually installed plugin."""
import importlib.util
import sys
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


def snapshot_path():
    path = Path(HF_HUB_CACHE) / "models--google--gemma-4-31B-it/snapshots" / REVISION
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return path


def plugin_root():
    spec = importlib.util.find_spec("vllm_tt_plugin")
    path = Path(spec.origin).resolve().parents[2]
    if not (path / ".git").exists() or not (path / "src/vllm_tt_plugin").is_dir():
        raise ValueError(f"Expected editable plugin checkout: {path}")
    return path


if __name__ == "__main__":
    print(snapshot_path() if sys.argv[1] == "snapshot" else plugin_root())
