# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh bring-up smoke test — the D3 prerequisite, run before any module work.

The target mesh opens at the spec's (8, 4), ``MeshConfig`` and ``CCLManager`` construct, and the
three collectives every module depends on round-trip: TP all-gather, TP all-reduce, and the SP
sequence shard/gather that carries the residual stream.

This is deliberately the FIRST device test: "worked on one card, broke on the mesh" is the bug class
that disappears if every module PCC test exercises sharding and collectives from the start, and this
is what proves the fabric is up before any of them run.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.tests.common import (
    galaxy_mesh,
    make_ccl,
    pcc,
    spec_mesh_config,
    sp_shard_activation,
    to_torch_replicated,
    to_torch_sp_concat,
)


@galaxy_mesh()
def test_mesh_opens_at_spec_shape(mesh_device, device_params, topology_name):
    assert tuple(mesh_device.shape) == (8, 4), f"opened {tuple(mesh_device.shape)}, spec binds (8, 4)"
    mc = spec_mesh_config(mesh_device)
    assert (mc.tp, mc.sp) == (4, 8)
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"mesh 8x4 open on {topology_name} fabric; compute grid {grid.x}x{grid.y}")
    assert grid.x >= 2, "the ring SDPA needs at least one compute column beside the CCL column"


@galaxy_mesh()
def test_tp_all_gather(mesh_device, device_params, topology_name):
    """Shard a width across the TP columns and gather it back."""
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    width = 1024
    host = torch.randn(1, 1, 128, width)
    tt = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mc.column_parallel(mesh_device),
    )
    gathered = mc.all_gather(tt, ccl, axis=mc.tp_axis, dim=3)
    got = to_torch_replicated(gathered, mesh_device)
    err = (got - host).abs().max().item()
    logger.info(f"[{topology_name}] TP all-gather max abs err {err:.5f}")
    assert pcc(host, got) > 0.999, "all-gather round-trip lost the data"


@galaxy_mesh()
def test_tp_all_reduce(mesh_device, device_params, topology_name):
    """Replicate a tensor across TP and all-reduce it: every column should see tp x the input."""
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    host = torch.randn(1, 1, 128, 1024)
    tt = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    reduced = mc.all_reduce(tt, ccl)
    got = to_torch_replicated(reduced, mesh_device)
    expected = host * mc.tp
    rel = ((got - expected).abs() / expected.abs().clamp(min=1e-3)).max().item()
    logger.info(f"[{topology_name}] TP all-reduce max rel err {rel:.5f}")
    assert pcc(expected, got) > 0.999


@galaxy_mesh()
def test_sp_sequence_roundtrip(mesh_device, device_params, topology_name):
    """The residual-stream layout: shard the sequence across SP rows, replicate across TP, gather back.

    A chunk of 5120 tokens gives 640 rows per SP row — the real per-device sequence length, so this
    also proves the shape divides the way the KV cache assumes.
    """
    mc = spec_mesh_config(mesh_device)
    chunk = 5120
    host = torch.randn(1, 1, chunk, 512)
    tt = sp_shard_activation(host, mesh_device, mc)
    assert tt.shape[-2] == chunk // mc.sp == 640
    got = to_torch_sp_concat(tt, mesh_device, mc)
    assert pcc(host, got) > 0.999


@galaxy_mesh()
def test_ccl_core_grid_leaves_room_for_sdpa(mesh_device, device_params):
    """The ring SDPA asserts its CCL offset is at or past the SDPA grid's x extent. Pin the
    invariant here so a change to either side fails in a one-line test, not inside an op."""
    from models.demos.llama_3_1_8b.tt.compute import ring_sdpa_program_config

    ccl = make_ccl(mesh_device)
    prog = ring_sdpa_program_config(mesh_device)
    assert ccl.ring_attention_ccl_core_grid_offset[0] >= prog.compute_with_storage_grid_size.x
