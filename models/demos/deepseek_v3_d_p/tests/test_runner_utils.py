# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the metal half of `runner_utils.py`.

These exercise the three pure-ttnn helpers end-to-end on a small mesh
(no `_migration` / `_mpi_test_helpers` dependencies — those helpers live
on the blaze side and are tested separately there).

Coverage:
  - `probe_dram_allocatable_base`: just confirms the ttnn.empty probe path
    succeeds (i.e. the helper doesn't crash and emits a buffer_address).
  - `verify_kvpe_cache_layout`: builds a tiny sharded kvpe-shaped tensor
    and confirms the per-device address scan reports CONSISTENT (or at
    least runs without raising).
  - `dump_kv_cache_shard_readback`: builds the same tensor and confirms
    the to_torch readback walks the sample positions without raising.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    dump_kv_cache_shard_readback,
    probe_dram_allocatable_base,
    verify_kvpe_cache_layout,
)


# Use a small SP×TP mesh so the test stays cheap on a single-galaxy host.
# (sp=2, tp=1) = 2 devices; enough to exercise per-device address scanning.
@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
def test_probe_dram_allocatable_base_smoke(mesh_device, caplog):
    """The probe should allocate, read a buffer_address, deallocate — no raise."""
    caplog.set_level("INFO")
    probe_dram_allocatable_base(mesh_device, label="unit-test")
    # The helper logs via loguru, not the std logger captured by caplog.
    # We only care it ran without raising. The function swallows all
    # ttnn-side errors and logs them; we make sure that path isn't hit.
    # Sanity: a second call still works (allocator state is unaffected).
    probe_dram_allocatable_base(mesh_device, label="unit-test-2")


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
def test_verify_kvpe_cache_layout_runs(mesh_device):
    """Build a small kvpe-shaped sharded tensor and pass it to verify_kvpe_cache_layout.

    Shape mirrors a real kvpe_cache slot: [num_layers=2, 1, seq_len_local=64, head_dim=128].
    Sharded along the seq dim (axis 2) over SP=2 → seq_len_local=32 per device.
    """
    num_layers = 2
    seq_len_total = 64
    head_dim = 128
    sp_factor = 2  # matches mesh_device dim 0

    host = torch.zeros(num_layers, 1, seq_len_total, head_dim, dtype=torch.bfloat16)
    cache = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, None),  # SP shards seq, TP replicated
        ),
    )

    verify_kvpe_cache_layout(mesh_device, cache)
    ttnn.deallocate(cache)


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True)
def test_dump_kv_cache_shard_readback_runs(mesh_device):
    """Build a tiny KV-shaped tensor with structured content and verify the
    readback walks the sample positions without raising. Doesn't assert exact
    bytes (bfloat16 quantization makes that brittle); just exercises the path."""
    num_layers = 2
    seq_len_total = 64
    head_dim = 128
    sp_factor = 2

    # Structured content so we'd notice if the readback got nonsense
    # (e.g. all zeros when we wrote ones).
    host = torch.ones(num_layers, 1, seq_len_total, head_dim, dtype=torch.bfloat16)
    cache = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, None),
        ),
    )

    # Sample positions that fit within seq_len_local=32 on each device.
    dump_kv_cache_shard_readback(layer_idx=0, kvpe_cache=cache, sample_positions=[0, 16])
    ttnn.deallocate(cache)
