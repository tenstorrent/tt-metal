# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy reader for the real Qwen3.8-27B safetensors checkpoint (torch + safetensors only).

The checkpoint is plain bf16 (no quantization, nothing to dequantize). Keys of the text model live
under ``model.language_model.``; ``lm_head.weight`` is top level; ``model.visual.*`` and ``mtp.*`` are
not part of the prefill text model and are never read. Every read fails loudly on a missing key.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

TEXT_PREFIX = "model.language_model."


def checkpoint_dir() -> Path:
    p = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
    if not p:
        raise RuntimeError("PREFILL_HF_MODEL / HF_MODEL is not set")
    return Path(p)


class CheckpointReader:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else checkpoint_dir()
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    @lru_cache(maxsize=32)
    def _file(self, fname):
        return safe_open(str(self.path / fname), framework="pt")

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"{key} not in checkpoint {self.path}")
        return self._file(self.weight_map[key]).get_tensor(key)

    def text(self, rel: str) -> torch.Tensor:
        """``rel`` relative to the text model, e.g. ``embed_tokens.weight`` / ``norm.weight``."""
        return self.get(TEXT_PREFIX + rel)

    def layer(self, i: int) -> dict[str, torch.Tensor]:
        """All tensors of decoder layer i, keyed relative to ``layers.{i}.`` (HF names)."""
        pre = f"{TEXT_PREFIX}layers.{i}."
        keys = sorted(k for k in self.weight_map if k.startswith(pre))
        if not keys:
            raise KeyError(f"no tensors for layer {i}")
        return {k[len(pre) :]: self.get(k) for k in keys}

    def lm_head(self) -> torch.Tensor:
        return self.get("lm_head.weight")
