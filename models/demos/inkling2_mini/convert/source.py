# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Inkling-2-Mini routed-expert dequantization (pure torch, no ttnn).

The checkpoint stores each expert tensor as ``[E, rows, K]`` (torch ``[out, in]`` per expert):

* NVFP4 (layers 0-19): ``w`` uint8 ``[E, rows, K/2]`` + ``w_scale`` float8_e4m3fn ``[E, rows, K/16]``
  + ``w_amax`` fp32 ``[E]`` + ``w_outer_scale_denominator`` fp32 ``[1]``:
  ``w = e2m1 * scale * amax[e] / denominator``.
* MXFP8 (layers 20-23): ``w`` float8_e4m3fn ``[E, rows, K]`` + ``w_scale`` uint8 UE8M0 ``[E, rows, K/32]``.

Same math as transformers' ``Inkling2LoopDequantizeExperts``; reuses the minimax_m3 per-matrix helpers.
``Checkpoint`` reads the safetensors shards by mmap, one tensor (or expert slice) at a time.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.minimax_m3.utils.checkpoint import dequant_mxfp8, dequant_nvfp4


def dequant_experts(
    weight: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor | None = None,
    denominator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a slice of experts ``[e, rows, *]`` to bf16 ``[e, rows, K]``. NVFP4 iff ``amax`` is given."""
    if amax is None:
        return torch.stack([dequant_mxfp8(weight[i], scale[i]) for i in range(weight.shape[0])])
    if denominator is None or amax.shape[0] != weight.shape[0]:
        raise ValueError(f"NVFP4 needs denominator and one amax per expert, got amax {tuple(amax.shape)}")
    global_scale = denominator.float().reshape(()) / amax.float()
    return torch.stack([dequant_nvfp4(weight[i], scale[i], global_scale[i]) for i in range(weight.shape[0])])


def deinterleave_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """``[e, 2I, K]`` with rows gate0, up0, gate1, up1, ... -> ``[e, 2I, K]`` rows gate 0:I, up I:2I
    (transformers ``Interleave(dim=1)``)."""
    e, rows, k = w13.shape
    return w13.view(e, rows // 2, 2, k).transpose(1, 2).reshape(e, rows, k)


class Checkpoint:
    """Read-only view of a (possibly sharded) safetensors checkpoint directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._files = {}
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:  # no index: every *.safetensors file in the directory (e.g. one file per tensor)
            self.weight_map = {
                k: f.name for f in sorted(self.root.glob("*.safetensors")) for k in self._open(f.name).keys()
            }

    def _open(self, shard: str):
        if shard not in self._files:
            self._files[shard] = safe_open(str(self.root / shard), framework="pt")
        return self._files[shard]

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        return self._open(self.weight_map[key]).get_tensor(key)

    def experts(self, key: str, start: int, end: int) -> torch.Tensor:
        """Experts ``start:end`` of an ``[E, ...]`` tensor, reading only those bytes."""
        return self._open(self.weight_map[key]).get_slice(key)[start:end]

    def header_sha256(self) -> str:
        """sha256 over each shard's safetensors header (length prefix + JSON), shards in name order."""
        h = hashlib.sha256()
        for shard in sorted(set(self.weight_map.values())):
            with open(self.root / shard, "rb") as f:
                prefix = f.read(8)
                h.update(prefix + f.read(struct.unpack("<Q", prefix)[0]))
        return h.hexdigest()
