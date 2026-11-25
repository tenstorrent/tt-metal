# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lazy host access to one layer's checkpoint tensors (fp8 groups dequantized to bf16), checkpoint naming."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import torch

_REF = Path(__file__).resolve().parents[2] / "reference"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

from checkpoint import (  # noqa: E402
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_layer_raw,
    load_tensors,
    shard_headers,
)

_EXPERT = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class LayerSource:
    """``src[name]`` -> the layer's raw checkpoint tensor; ``src.experts()`` -> stacked routed experts
    ``{"gate_up": [E, D, 2I] ([gate | up] columns), "down": [E, I, D]}``. Nothing is read until first use, so a warm
    weight cache never touches the checkpoint. ``raw`` serves an in-memory dict instead (tests)."""

    def __init__(self, layer_idx: int, snapshot: Optional[Path] = None, headers=None, raw: Optional[dict] = None):
        self.layer_idx = layer_idx
        self.snapshot = snapshot
        self.headers = headers
        self._raw = None if raw is None else {k: v for k, v in raw.items() if not _EXPERT.match(k)}
        self._raw_experts = None if raw is None else {k: v for k, v in raw.items() if _EXPERT.match(k)}
        self._experts = None

    def __getitem__(self, name: str) -> torch.Tensor:
        if self._raw is None:
            self._raw = load_layer_raw(
                self.snapshot, self.layer_idx, self.headers, name_filter=lambda n: not n.startswith("mlp.experts.")
            )
        return self._raw[name]

    def experts(self) -> dict:
        if self._experts is None:
            raw = self._raw_experts
            if raw is None:
                raw = load_layer_raw(
                    self.snapshot, self.layer_idx, self.headers, name_filter=lambda n: n.startswith("mlp.experts.")
                )
            per = {}
            for k, t in raw.items():
                m = _EXPERT.match(k)
                per.setdefault(int(m.group(1)), {})[m.group(2)] = t
            n = len(per)
            assert sorted(per) == list(range(n)), "expert index gap"
            self._experts = {
                "gate_up": torch.stack(
                    [torch.cat([per[e]["gate_proj"].t(), per[e]["up_proj"].t()], 1) for e in range(n)]
                ),
                "down": torch.stack([per[e]["down_proj"].t() for e in range(n)]),
            }
            self._raw_experts = None
        return self._experts

    def release(self):
        self._raw = self._raw_experts = self._experts = None


class ModelSource:
    def __init__(self, snapshot: Optional[Path] = None):
        self.snapshot = Path(snapshot) if snapshot is not None else default_snapshot_dir()
        self._headers = None

    @property
    def headers(self):
        if self._headers is None:
            self._headers = shard_headers(self.snapshot)
        return self._headers

    def layer(self, idx: int) -> LayerSource:
        return LayerSource(idx, self.snapshot, self.headers)

    def tensor(self, key: str) -> torch.Tensor:
        return load_tensors(self.snapshot, [key], self.headers)[key]

    def embedding(self):
        return self.tensor(EMBED_KEY)

    def final_norm(self):
        return self.tensor(FINAL_NORM_KEY)

    def lm_head(self):
        return self.tensor(LM_HEAD_KEY)
