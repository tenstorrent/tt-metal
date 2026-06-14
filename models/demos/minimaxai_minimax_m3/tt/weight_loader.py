# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy, single-tensor weight loader for the 869GB MiniMax-M3 checkpoint.

The full checkpoint is 23,416 tensors / 869 GB bf16 across 59 safetensors
shards. NEVER ``from_pretrained`` the whole model. This loader reads exactly
ONE named tensor at a time directly from the shard that holds it, using the
``model.safetensors.index.json`` ``weight_map`` and ``safetensors.safe_open``
(mmap, slice-on-read), then converts it to a ttnn mesh tensor with the
TP=32 sharding-recipe mesh-mapper picked for that weight.

Key prefixes (confirmed in the index):
  text decoder : ``language_model.model.*``
  lm_head      : ``language_model.lm_head.weight``
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.demos.minimaxai_minimax_m3.tt import model_config as mc


def find_model_path() -> Path:
    """Locate the MiniMax-M3 HF snapshot dir (the one holding the index)."""
    env = os.environ.get("MINIMAX_M3_PATH")
    if env and (Path(env) / "model.safetensors.index.json").is_file():
        return Path(env)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base = Path(hf_home) / "hub" / "models--MiniMaxAI--MiniMax-M3" / "snapshots"
    if base.is_dir():
        for snap in sorted(base.iterdir()):
            if (snap / "model.safetensors.index.json").is_file():
                return snap
    raise FileNotFoundError("Could not locate the MiniMax-M3 snapshot. Set MINIMAX_M3_PATH or HF_HOME.")


@lru_cache(maxsize=1)
def _weight_map(model_path: str) -> dict:
    """Parse ``model.safetensors.index.json`` -> {tensor_name: shard_file}."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        return dict(json.load(f)["weight_map"])


# Cache open safetensors handles per shard so repeated reads from the same
# shard do not re-mmap. Handles are torch-framework slices.
_HANDLE_CACHE: dict = {}


def _handle(model_path: Path, shard_file: str):
    key = (str(model_path), shard_file)
    h = _HANDLE_CACHE.get(key)
    if h is None:
        h = safe_open(str(Path(model_path) / shard_file), framework="pt", device="cpu")
        _HANDLE_CACHE[key] = h
    return h


def load_torch_weight(key: str, model_path: Path | None = None) -> torch.Tensor:
    """Load exactly ONE named tensor from its shard. No full-model load."""
    model_path = Path(model_path) if model_path is not None else find_model_path()
    wmap = _weight_map(str(model_path))
    if key not in wmap:
        raise KeyError(f"Weight {key!r} not in index ({len(wmap)} tensors).")
    return _handle(model_path, wmap[key]).get_tensor(key)


def _resolve_mapper(mesh, shard: str, dim: int | None):
    """Pick the mesh-mapper for a weight per the TP=32 recipe (model_config)."""
    if shard == "replicate":
        return mc.replicate_mapper(mesh)
    if shard == "column":  # shard output dim -1
        return mc.column_parallel_mapper(mesh)
    if shard == "row":  # shard input dim -2
        return mc.row_parallel_mapper(mesh)
    if shard == "vocab":  # lm_head output vocab dim -1
        return mc.vocab_shard_mapper(mesh)
    if shard == "dim":
        assert dim is not None, "shard='dim' needs an explicit dim"
        return mc.shard_dim_mapper(mesh, dim)
    raise ValueError(f"Unknown shard recipe {shard!r}")


def to_mesh_tensor(
    torch_weight: torch.Tensor,
    mesh,
    *,
    shard: str = "replicate",
    dim: int | None = None,
    dtype=mc.WEIGHT_DTYPE,
    layout=ttnn.TILE_LAYOUT,
    add_gemma_one: bool = False,
) -> ttnn.Tensor:
    """Convert a torch weight to a ttnn mesh tensor with the right mapper.

    Args:
        shard: one of ``replicate`` | ``column`` | ``row`` | ``vocab`` | ``dim``.
        add_gemma_one: bake the gemma ``+1`` into the gamma host-side so the
            on-device rms_norm forward stays a single fused op (no add).
    """
    w = torch_weight
    if add_gemma_one:
        # gemma RMSNorm scales by (1 + weight); fold the +1 into the gamma so
        # ttnn.rms_norm's plain multiply reproduces it exactly.
        w = w.float() + 1.0
    mapper = _resolve_mapper(mesh, shard, dim)
    return ttnn.from_torch(w, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper)


def load_weight_to_mesh(
    key: str,
    mesh,
    *,
    shard: str = "replicate",
    dim: int | None = None,
    dtype=mc.WEIGHT_DTYPE,
    layout=ttnn.TILE_LAYOUT,
    add_gemma_one: bool = False,
    model_path: Path | None = None,
) -> ttnn.Tensor:
    """Load one named weight and push it to the mesh with its TP recipe."""
    w = load_torch_weight(key, model_path=model_path)
    return to_mesh_tensor(w, mesh, shard=shard, dim=dim, dtype=dtype, layout=layout, add_gemma_one=add_gemma_one)
