# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the device PCC tests: spec thresholds, and host <-> mesh layouts.

Layouts on the 8x4 mesh (rows = SP, cols = TP):
  * residual  ``[1, 1, S, H]``    seq over SP rows, hidden over TP cols      (``[1, 1, S/8, H/4]`` per chip)
  * full      ``[1, 1, S, H]``    seq over SP rows, hidden replicated on TP  (``[1, 1, S/8, H]`` per chip)
  * heads     ``[1, NH, S, D]``   heads over TP cols, seq over SP rows       (``[1, NH/4, S/8, D]`` per chip)
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.config import (
    MistralMediumConfig,
    pcc_thresholds,
    resolve_dataformats,
    ttnn_dtype,
)

CFG = MistralMediumConfig.from_json()
PCC_TARGET, PCC_LOWER_BOUND = pcc_thresholds()


def spec_dtypes() -> dict:
    return {k: ttnn_dtype(v) for k, v in resolve_dataformats().items()}


def assert_pcc(name, ref, got, lower=PCC_LOWER_BOUND):
    """Assert the spec's pcc_lower_bound; log the value and flag anything under pcc_target."""
    assert ref.shape == got.shape, f"{name}: shape {tuple(got.shape)} != reference {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref.float(), got.float(), lower)
    note = "" if pcc >= PCC_TARGET else f"  [below pcc_target {PCC_TARGET}]"
    logger.info(f"[pcc] {name}: {pcc:.6f}{note}")
    assert passing, f"{name}: PCC {pcc} < pcc_lower_bound {lower}"
    return pcc


def _to_mesh(x, mesh, mesh_config, sp_dim, tp_dim, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        x,
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.mapper(mesh, sp_dim=sp_dim, tp_dim=tp_dim),
    )


def to_residual(x, mesh, mesh_config, dtype=ttnn.bfloat16):
    return _to_mesh(x, mesh, mesh_config, sp_dim=2, tp_dim=3, dtype=dtype)


def to_full(x, mesh, mesh_config, dtype=ttnn.bfloat16):
    return _to_mesh(x, mesh, mesh_config, sp_dim=2, tp_dim=None, dtype=dtype)


def to_heads(x, mesh, mesh_config, dtype=ttnn.bfloat16):
    return _to_mesh(x, mesh, mesh_config, sp_dim=2, tp_dim=1, dtype=dtype)


def _compose(tt, mesh, mesh_config, sp_dim, tp_dim):
    dims = [None, None]
    dims[mesh_config.sp_axis], dims[mesh_config.tp_axis] = sp_dim, tp_dim
    return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=mesh.shape, dims=dims)).float()


def residual_to_torch(tt, mesh, mesh_config):
    return _compose(tt, mesh, mesh_config, sp_dim=2, tp_dim=3)


def heads_to_torch(tt, mesh, mesh_config):
    return _compose(tt, mesh, mesh_config, sp_dim=2, tp_dim=1)


def full_to_torch(tt, mesh, mesh_config, check_replicated=True):
    """TP-replicated tensor -> host (TP column 0); optionally assert every column holds the same data."""
    tp = mesh_config.tp
    per = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tt)]
    rows = [per[r * tp] for r in range(mesh_config.sp)]
    if check_replicated:
        for r in range(mesh_config.sp):
            for c in range(1, tp):
                assert torch.equal(per[r * tp + c], rows[r]), f"TP column {c} of row {r} diverged"
    return torch.cat(rows, dim=-2)


def randn(*shape, seed=0, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale).to(dtype)
