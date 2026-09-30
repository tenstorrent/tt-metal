# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for Xing4.0-29B-A4B (XingChen-AGI/Xing4.0-29B-A4B @ baae3c3e).

Storage: every weight BF16 except ``*.hc_scale`` and ``mlp.gate.e_score_correction_bias`` (F32). No quantization:
the "dequantization" is a dtype cast. Routed experts are stored per expert (``mlp.experts.{e}.{gate,up,down}_proj``).
The MTP layer (``model.layers.40.*``) is out of scope and never read.

Every tensor is looked up through ``model.safetensors.index.json`` (never by shard name): R.4 trims the checkpoint to
the subset's layers and renames mixed shards.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

MTP_PREFIX = "model.layers.40."


class WeightLoader:
    """Lazy safetensors accessor keyed by checkpoint tensor name."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def get(self, name: str) -> torch.Tensor:
        assert not name.startswith(MTP_PREFIX), f"MTP is out of scope: {name}"
        fname = os.path.join(self.model_path, self.weight_map[name])
        if fname not in self._handles:
            self._handles[fname] = safe_open(fname, framework="pt")
        return self._handles[fname].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map
