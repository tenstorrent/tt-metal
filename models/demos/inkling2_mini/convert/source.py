# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Inkling-2-Mini routed-expert dequantization (pure torch, no ttnn).

The checkpoint stores each expert tensor as ``[E, rows, K]`` (torch ``[out, in]`` per expert):

* NVFP4 (layers 0-19): ``w`` uint8 ``[E, rows, K/2]`` + ``w_scale`` float8_e4m3fn ``[E, rows, K/16]``
  + ``w_amax`` fp32 ``[E]`` + ``w_outer_scale_denominator`` fp32 ``[1]``:
  ``w = e2m1 * scale * amax[e] / denominator``.
* MXFP8 (layers 20-23): ``w`` float8_e4m3fn ``[E, rows, K]`` + ``w_scale`` uint8 UE8M0 ``[E, rows, K/32]``.

Same math as transformers' ``Inkling2LoopDequantizeExperts``. The per-matrix helpers ``dequant_mxfp8`` /
``dequant_nvfp4`` are copied from models/demos/minimax_m3/utils/checkpoint.py, so a change there cannot change
this converter.
``Checkpoint`` reads the safetensors shards by mmap, one tensor (or expert slice) at a time.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

MXFP8_BLOCK_SIZE = 32
NVFP4_GROUP_SIZE = 16
# E2M1 code -> value; bit 3 is the sign.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_mxfp8(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = MXFP8_BLOCK_SIZE) -> torch.Tensor:
    """float8_e4m3fn ``[N, K]`` + UE8M0 uint8 ``[N, K/block]`` -> bf16 ``[N, K]``."""
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"MXFP8 weight must be float8_e4m3fn, got {weight.dtype}")
    if scale_inv.dtype != torch.uint8:
        raise ValueError(f"MXFP8 scale must be uint8 (UE8M0), got {scale_inv.dtype}")
    n, k = weight.shape
    if k % block_size or tuple(scale_inv.shape) != (n, k // block_size):
        raise ValueError(f"MXFP8 scale shape {tuple(scale_inv.shape)} does not match weight {(n, k)} / {block_size}")
    scale = torch.exp2(scale_inv.to(torch.float32) - 127.0).unsqueeze(-1)
    out = weight.to(torch.float32).view(n, k // block_size, block_size) * scale
    return out.view(n, k).to(torch.bfloat16)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 ``[..., K/2]`` -> fp32 ``[..., K]`` (low nibble first)."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed E2M1 must be uint8, got {packed.dtype}")
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    return table[codes.long()]


def dequant_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
) -> torch.Tensor:
    """uint8 ``[N, K/2]`` + float8_e4m3fn ``[N, K/group]`` + fp32 scalar -> bf16 ``[N, K]``."""
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"NVFP4 block scale must be float8_e4m3fn, got {scale.dtype}")
    if global_scale.numel() != 1:
        raise ValueError(f"NVFP4 global scale must be a scalar, got shape {tuple(global_scale.shape)}")
    n, k = packed.shape[0], packed.shape[1] * 2
    if k % group_size or tuple(scale.shape) != (n, k // group_size):
        raise ValueError(f"NVFP4 scale shape {tuple(scale.shape)} does not match weight {(n, k)} / {group_size}")
    values = unpack_e2m1(packed).view(n, k // group_size, group_size)
    block_scale = scale.to(torch.float32).unsqueeze(-1) / global_scale.to(torch.float32).reshape(())
    return (values * block_scale).view(n, k).to(torch.bfloat16)


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
