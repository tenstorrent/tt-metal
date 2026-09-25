# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lazy safetensors loading of the text sub-tree of a Gemma-4 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import DEFAULT_CKPT_DIR

TEXT_PREFIX = "model.language_model."


class CheckpointReader:
    def __init__(self, ckpt_dir: str | Path = DEFAULT_CKPT_DIR):
        self.dir = Path(ckpt_dir)
        idx = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self.weight_map = {k[len(TEXT_PREFIX) :]: v for k, v in idx.items() if k.startswith(TEXT_PREFIX)}
        self._handles = {}

    def _handle(self, fname):
        if fname not in self._handles:
            self._handles[fname] = safe_open(str(self.dir / fname), framework="pt")
        return self._handles[fname]

    def get(self, key: str) -> torch.Tensor:
        return self._handle(self.weight_map[key]).get_tensor(TEXT_PREFIX + key)

    def keys(self, prefix: str = ""):
        return [k for k in self.weight_map if k.startswith(prefix)]

    def substate(self, prefix: str) -> dict[str, torch.Tensor]:
        p = prefix if prefix.endswith(".") or not prefix else prefix + "."
        return {k[len(p) :]: self.get(k) for k in self.keys(p)}

    def layer_state(self, i: int) -> dict[str, torch.Tensor]:
        return self.substate(f"layers.{i}")


def load_text_model(model, reader: CheckpointReader, layers: list[int] | None = None, dtype=torch.bfloat16):
    """Fill ``Gemma4TextModel`` (possibly truncated) from the checkpoint; returns the model."""
    model.embed_tokens.weight.data = reader.get("embed_tokens.weight").to(dtype)
    model.norm.weight.data = reader.get("norm.weight").to(dtype)
    if model.cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    for i, layer in enumerate(model.layers):
        src = layers[i] if layers else i
        missing, unexpected = layer.load_state_dict(
            {k: v.to(dtype) for k, v in reader.layer_state(src).items()}, strict=False
        )
        assert not unexpected, f"unexpected keys layer {src}: {unexpected}"
        assert all(m.startswith("self_attn.v_norm") or m == "self_attn.v_proj.weight" for m in missing), missing
    return model.to(dtype)
