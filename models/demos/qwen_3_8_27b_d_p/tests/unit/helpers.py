# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the device PCC tests.

Every test here follows one shape: build the **torch reference** module with seeded random
weights, hand the *same* state dict to the TT module, push the same activation through both, and
PCC. The reference computes in fp16 (recipe section 4); the device runs the spec's dtypes, and the
gap between them is the number the PCC table records.
"""

from __future__ import annotations

from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ...config import MeshConfig
from ...reference.modeling import REF_DTYPE
from ...spec import load_spec, ttnn_dtype

SPEC = load_spec()
WEIGHT_DTYPE = ttnn_dtype(SPEC.weight_dtype)
ACTIVATION_DTYPE = ttnn_dtype(SPEC.activation_dtype)
CACHE_DTYPE = ttnn_dtype(SPEC.kv_cache_dtype)


def randn(*shape: int, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(REF_DTYPE)


def to_replicated(x: torch.Tensor, mesh, mesh_config: MeshConfig, dtype=None) -> ttnn.Tensor:
    return ttnn.from_torch(
        x.float(),
        dtype=dtype or ACTIVATION_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.replicate(mesh),
    )


def to_sp_sharded(x: torch.Tensor, mesh, mesh_config: MeshConfig, *, seq_dim: int = 2, dtype=None) -> ttnn.Tensor:
    """Sequence-shard across the SP rows, replicate across the TP cols — the residual's layout."""
    return ttnn.from_torch(
        x.float(),
        dtype=dtype or ACTIVATION_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.sequence_parallel(mesh, seq_dim=seq_dim),
    )


def from_sp_sharded(tt: ttnn.Tensor, mesh_config: MeshConfig, *, seq_dim: int = 2) -> torch.Tensor:
    """Concatenate the SP rows' shards back into the full sequence (column 0 of each row).

    Row-major device order: device ``r * tp`` is row ``r``, column 0. Taking column 0 rather than
    composing all columns is deliberate — for a TP-replicated activation the columns must agree,
    and :func:`assert_tp_replicated` is the test that checks they do.
    """
    shards = ttnn.get_device_tensors(tt)
    rows = [ttnn.to_torch(shards[r * mesh_config.tp]) for r in range(mesh_config.sp)]
    return torch.cat(rows, dim=seq_dim)


def from_tp_sharded(tt: ttnn.Tensor, mesh_config: MeshConfig, *, dim: int = -1) -> torch.Tensor:
    """Concatenate the TP columns' shards of row 0."""
    shards = ttnn.get_device_tensors(tt)
    return torch.cat([ttnn.to_torch(shards[c]) for c in range(mesh_config.tp)], dim=dim)


def from_sp_tp_sharded(tt: ttnn.Tensor, mesh_config: MeshConfig, *, seq_dim: int = 2, tp_dim: int = -1):
    """Compose a tensor sharded on BOTH axes: sequence on the rows, features on the cols."""
    shards = ttnn.get_device_tensors(tt)
    rows = []
    for r in range(mesh_config.sp):
        cols = [ttnn.to_torch(shards[r * mesh_config.tp + c]) for c in range(mesh_config.tp)]
        rows.append(torch.cat(cols, dim=tp_dim))
    return torch.cat(rows, dim=seq_dim)


def assert_tp_replicated(tt: ttnn.Tensor, mesh_config: MeshConfig, name: str, pcc: float = 0.999) -> None:
    """Every TP column of row 0 must hold the same values — the residual-stream invariant."""
    if mesh_config.tp == 1:
        return
    shards = ttnn.get_device_tensors(tt)
    ref = ttnn.to_torch(shards[0]).float()
    for c in range(1, mesh_config.tp):
        ok, value = comp_pcc(ref, ttnn.to_torch(shards[c]).float(), pcc)
        assert ok, f"{name}: TP column {c} diverged from column 0 ({value}) — the residual is not replicated"


def check_pcc(name: str, expected: torch.Tensor, actual: torch.Tensor, *, shape=None) -> float:
    """Assert at the spec's ``pcc_lower_bound`` and log the value against ``pcc_target``.

    Returns the measured PCC so a caller can record it in the README table.
    """
    if shape is not None:
        actual = actual.reshape(shape)
        expected = expected.reshape(shape)
    assert expected.shape == actual.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    lower, target = SPEC.acceptance.pcc_lower_bound, SPEC.acceptance.pcc_target
    passing, value = comp_pcc(expected.float(), actual.float(), lower)
    verdict = "at target" if float(value) >= target else f"BELOW target {target}"
    logger.info(f"PCC {name}: {value} ({verdict}, lower bound {lower})")
    assert passing, f"{name} PCC {value} is below the spec's lower bound {lower}"
    return float(value)


def sp_chunk(mesh_config: MeshConfig, per_row_tokens: int = 128) -> int:
    """A chunk length that splits cleanly across SP and is tile-aligned per row."""
    assert per_row_tokens % ttnn.TILE_SIZE == 0
    return per_row_tokens * mesh_config.sp


def deallocate(*tensors: Optional[ttnn.Tensor]) -> None:
    for t in tensors:
        if t is not None:
            t.deallocate(True)
