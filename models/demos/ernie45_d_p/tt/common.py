# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the ERNIE-4.5 TTNN prefill implementation (mesh 1x4, TP/EP = 4)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

import ttnn

REPO = Path(__file__).resolve().parents[4]
CACHE_ROOT = Path(os.environ.get("ERNIE_TT_CACHE", REPO / "generated/ernie45_d_p/tt_cache"))
GOLDEN_ROOT = Path(os.environ.get("ERNIE_GOLDEN_ROOT", REPO / "generated/ernie45_d_p/golden"))

# ERNIE prefill needs fp32 accumulation for long-K matmuls and softmax-style reductions.
COMPUTE_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)

DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "l1_small_size": 24576}


def cache_name(*parts) -> str | None:
    if os.environ.get("ERNIE_NO_TT_CACHE"):
        return None
    p = CACHE_ROOT.joinpath(*[str(x) for x in parts])
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def replicate(mesh, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, cache=None, mem=None):
    return ttnn.as_tensor(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache,
    )


def shard(mesh, t: torch.Tensor, dim: int, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, cache=None, mem=None):
    return ttnn.as_tensor(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache,
    )


def to_mesh_activation(mesh, x: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[S, H] host activation -> replicated [1, 1, S, H] tiled device tensor."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def device_tensors(t: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


def replicated_to_torch(t: ttnn.Tensor, check_equal: bool = False) -> torch.Tensor:
    parts = device_tensors(t)
    if check_equal:
        for p in parts[1:]:
            assert torch.equal(p, parts[0]), "replicated tensor differs across devices"
    return parts[0]


def concat_to_torch(t: ttnn.Tensor, dim: int) -> torch.Tensor:
    return torch.cat(device_tensors(t), dim=dim)


class Golden:
    """Reader for generated/ernie45_d_p/golden/s{seq}_c{chunk} (see reference/generate_golden.py)."""

    def __init__(self, seq: int, chunk: int, root: Path | None = None):
        self.dir = (root or GOLDEN_ROOT) / f"s{seq}_c{chunk}"
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"golden not generated: {self.dir} (run the P1.4-P1.6 gates)")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.seq, self.chunk = seq, chunk
        self._cache = {}

    def layer(self, c: int, i: int) -> dict[str, torch.Tensor]:
        key = ("L", c, i)
        if key not in self._cache:
            self._cache[key] = load_file(str(self.dir / f"chunk_{c:02d}" / f"layer_{i:02d}.safetensors"))
        return self._cache[key]

    def model(self, c: int) -> dict[str, torch.Tensor]:
        return load_file(str(self.dir / f"chunk_{c:02d}" / "model.safetensors"))

    def kv(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = load_file(str(self.dir / "kv_cache" / f"layer_{i}.safetensors"))
        return d[f"key_cache_layer_{i}"][0], d[f"value_cache_layer_{i}"][0]  # [n_kv, seq, D]

    def tokens(self) -> torch.Tensor:
        return torch.tensor(json.loads((self.dir / "metadata.json").read_text())["token_ids"])
