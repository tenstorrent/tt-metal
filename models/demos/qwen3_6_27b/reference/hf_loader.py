# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HF weight loading helpers for Qwen3.6-27B."""
import json
import os
from pathlib import Path

from safetensors.torch import load_file as load_safetensors

DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def load_qwen36_index(snapshot_dir=DEFAULT_SNAPSHOT):
    idx_path = Path(snapshot_dir) / "model.safetensors.index.json"
    return json.load(open(idx_path))


def load_qwen36_tensors(keys, snapshot_dir=DEFAULT_SNAPSHOT):
    """Load named tensors lazily from safetensors shards."""
    idx = load_qwen36_index(snapshot_dir)
    weight_map = idx["weight_map"]
    files_needed = sorted({weight_map[k] for k in keys if k in weight_map})
    out = {}
    for fn in files_needed:
        shard = load_safetensors(str(Path(snapshot_dir) / fn))
        for k in keys:
            if k in shard:
                out[k] = shard[k]
    missing = set(keys) - set(out.keys())
    if missing:
        raise KeyError(f"missing keys: {sorted(missing)[:5]}...")
    return out


def load_qwen36_config(snapshot_dir=DEFAULT_SNAPSHOT):
    """Load config.json and override model_type to qwen3_next for HF compat (same arch)."""
    cfg_path = Path(snapshot_dir) / "config.json"
    cfg_dict = json.load(open(cfg_path))
    # Qwen3.6 = Qwen3-Next architecturally; rename for HF loading
    cfg_dict["model_type"] = "qwen3_next"
    if "text_config" in cfg_dict:
        cfg_dict["text_config"]["model_type"] = "qwen3_next"
    # Drop vision_config — qwen3_next doesn't have it (text path only for now)
    cfg_dict.pop("vision_config", None)
    return cfg_dict
