# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint discovery and loading for HuggingFace Time Series Transformer weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


def resolve_checkpoint_dir(checkpoint: str) -> Path:
    """Return a local directory for ``checkpoint``, downloading from the Hub if needed."""
    path = Path(checkpoint)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            checkpoint,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json", "*.bin"],
        )
    )


def load_checkpoint_state(
    checkpoint: str, *, key_prefixes: Optional[tuple[str, ...]] = None
) -> dict[str, torch.Tensor]:
    """Read a checkpoint's tensors, preferring safetensors over a pickled ``pytorch_model.bin``."""
    directory = resolve_checkpoint_dir(checkpoint)
    prefixes = tuple(key_prefixes or ())

    def wanted(key: str) -> bool:
        return not prefixes or key.startswith(prefixes)

    index_path = directory / "model.safetensors.index.json"
    single_path = directory / "model.safetensors"

    if index_path.is_file() or single_path.is_file():
        from safetensors import safe_open

        state: dict[str, torch.Tensor] = {}
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                weight_map: dict[str, str] = json.load(handle)["weight_map"]
            shards: dict[str, list[str]] = {}
            for key, filename in weight_map.items():
                if wanted(key):
                    shards.setdefault(filename, []).append(key)
        else:
            shards = {single_path.name: None}

        for filename, keys in shards.items():
            with safe_open(str(directory / filename), framework="pt", device="cpu") as handle:
                selected = keys if keys is not None else [k for k in handle.keys() if wanted(k)]
                for key in selected:
                    state[key] = handle.get_tensor(key)
        return state

    legacy_path = directory / "pytorch_model.bin"
    if legacy_path.is_file():
        state = torch.load(legacy_path, map_location="cpu", weights_only=True)
        return {key: value for key, value in state.items() if wanted(key)}

    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {directory}.")


def load_checkpoint_config(checkpoint: str):
    """Load the HuggingFace config object for ``checkpoint``."""
    from transformers import TimeSeriesTransformerConfig as HFConfig

    directory = resolve_checkpoint_dir(checkpoint)
    return HFConfig.from_pretrained(str(directory))


def extract_embedder_weights(state: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Pull the static categorical embedding tables out of a state dict, in column order."""
    weights: list[torch.Tensor] = []
    index = 0
    while True:
        key = f"model.embedder.embedders.{index}.weight"
        if key not in state:
            break
        weights.append(state[key].detach().float())
        index += 1
    return weights


__all__ = [
    "extract_embedder_weights",
    "load_checkpoint_config",
    "load_checkpoint_state",
    "resolve_checkpoint_dir",
]
