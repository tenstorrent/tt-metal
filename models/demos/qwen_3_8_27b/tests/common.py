# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers: PCC, the spec's thresholds, SP/TP host<->device helpers."""

import torch
from loguru import logger

import ttnn
from models.demos.qwen_3_8_27b.config import PrefillSpec


def pcc(a, b) -> float:
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def assert_pcc(name, got, want, spec: PrefillSpec | None = None):
    """Assert at the spec's pcc_lower_bound; log (and return) the value, flagging anything under pcc_target."""
    spec = spec or PrefillSpec.load()
    p = pcc(got.float(), want.float())
    tag = "OK" if p >= spec.pcc_target else ("BELOW_TARGET" if p >= spec.pcc_lower_bound else "FAIL")
    logger.info(f"PCC[{name}] = {p:.6f} ({tag}; target {spec.pcc_target}, lower bound {spec.pcc_lower_bound})")
    print(f"PCC[{name}] = {p:.6f} ({tag})")
    assert p == p and p >= spec.pcc_lower_bound, f"{name}: PCC {p:.6f} < pcc_lower_bound {spec.pcc_lower_bound}"
    return p


def to_sp(x, mesh_config, dtype=ttnn.bfloat16, seq_dim=2, tp_dim=None, layout=ttnn.TILE_LAYOUT):
    """Host tensor -> device, sequence SP-sharded (contiguous per row), optionally TP-sharded on tp_dim."""
    return ttnn.from_torch(
        x,
        device=mesh_config.mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.shard(seq_dim, tp_dim),
    )


def from_sp(t, mesh_config, seq_dim=2, tp_dim=None):
    """Device SP-sharded tensor -> host. With tp_dim None, the TP-replicated column 0 is returned."""
    if tp_dim is not None:
        return ttnn.to_torch(t, mesh_composer=mesh_config.compose(seq_dim, tp_dim)).float()
    full = ttnn.to_torch(t, mesh_composer=mesh_config.compose(seq_dim, len(t.shape) - 1)).float()
    w = full.shape[-1] // mesh_config.tp
    return full[..., :w]
