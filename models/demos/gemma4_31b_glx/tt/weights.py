# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for the device modules: config values and single tensors read straight from the HF safetensors."""

import functools
import json
from pathlib import Path

import torch
from safetensors import safe_open

PREFIX = "model.language_model."


@functools.cache
def text_config(ckpt: str) -> dict:
    c = json.loads((Path(ckpt) / "config.json").read_text())
    return c.get("text_config", c)


@functools.cache
def _weight_map(ckpt: str) -> dict:
    return json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())["weight_map"]


def load(ckpt: str, name: str) -> torch.Tensor:
    """One tensor by its name under model.language_model (e.g. ``layers.0.input_layernorm.weight``)."""
    key = PREFIX + name
    with safe_open(str(Path(ckpt) / _weight_map(ckpt)[key]), framework="pt") as f:
        return f.get_tensor(key)
