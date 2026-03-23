# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Foundation tests for SubDevice-aware ops on Galaxy Wormhole TG mesh.

Validates that TTNN ops work correctly when a SubDevice manager restricts
compute to a worker-grid subset of the full 8x9 Tensix grid.

Galaxy WH grid layout (per chip):
  - Full grid: 8 cols (x=0..7) x 9 rows (y=0..8) = 72 cores

Two SubDevice layouts tested:
  1. "origin-worker": Worker on cols 0-5, sender on cols 6-7
     - Compatible with matmul's compute_with_storage_grid_size (starts from 0,0)
     - Proves that SubDevice + matmul works when grid is contiguous from origin
  2. "realistic": Worker on cols 1-6, sender on cols 0 and 7
     - Matches actual Galaxy CCL layout (ethernet dispatch on edge columns)
     - Matmul with default program_config FAILS here (grid starts at 0,0, spans both SubDevices)
     - This failure is the motivation for the SubDevice-aware op migration

Usage:
  cd /tt-metal
  python -m pytest tests/ttnn/unit_tests/operations/test_sub_device_ops.py -v -s
"""

import pytest
import torch
import ttnn


# ── Galaxy WH grid constants ──────────────────────────────────────────────────
GRID_X = 8  # columns 0..7
GRID_Y = 9  # rows 0..8


def make_origin_worker_sub_devices():
    """
    Worker cols 0-5, sender cols 6-7.
    Worker grid is contiguous from origin — compatible with matmul's
    compute_with_storage_grid_size which always starts from (0,0).
    """
    worker_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(5, GRID_Y - 1),
            ),
        }
    )
    sender_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(6, 0),
                ttnn.CoreCoord(7, GRID_Y - 1),
            ),
        }
    )
    return ttnn.SubDevice([worker_cores]), ttnn.SubDevice([sender_cores])


def make_realistic_sub_devices():
    """
    Worker cols 1-6, sender cols 0 and 7.
    Matches actual Galaxy CCL layout where ethernet dispatch uses edge columns.
    """
    worker_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(1, 0),
                ttnn.CoreCoord(6, GRID_Y - 1),
            ),
        }
    )
    sender_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, GRID_Y - 1),
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(7, 0),
                ttnn.CoreCoord(7, GRID_Y - 1),
            ),
        }
    )
    return ttnn.SubDevice([worker_cores]), ttnn.SubDevice([sender_cores])


# ── Test: SubDevice manager lifecycle on TG mesh ─────────────────────────────
@pytest.mark.parametrize("layout", ["origin", "realistic"])
def test_sub_device_manager_lifecycle(mesh_device, layout):
    """Verify SubDevice manager can be created, loaded, and removed on TG mesh."""
    if layout == "origin":
        worker_sd, sender_sd = make_origin_worker_sub_devices()
    else:
        worker_sd, sender_sd = make_realistic_sub_devices()

    mgr = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 3200)
    mesh_device.load_sub_device_manager(mgr)

    ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
    ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(1)])

    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    ttnn.synchronize_device(mesh_device)
    mesh_device.reset_sub_device_stall_group()

    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(mgr)


# ── Test: ttnn.add on DRAM-interleaved tensors with SubDevice ────────────────
@pytest.mark.parametrize("layout", ["origin", "realistic"])
def test_add_dram_interleaved_with_sub_device(mesh_device, layout):
    """
    ttnn.add on DRAM-interleaved tensors should work with SubDevice manager
    active. Eltwise ops use a single core and don't multicast, so they work
    with any SubDevice layout.
    """
    if layout == "origin":
        worker_sd, sender_sd = make_origin_worker_sub_devices()
    else:
        worker_sd, sender_sd = make_realistic_sub_devices()

    mgr = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 3200)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    num_devices = mesh_device.get_num_devices()
    shape = [1, 1, 32, 64]

    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = a_torch + b_torch

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    a_tt = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    b_tt = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    c_tt = ttnn.add(a_tt, b_tt)
    c_torch = ttnn.to_torch(c_tt, mesh_composer=mesh_composer)

    for i in range(num_devices):
        assert torch.allclose(c_torch[i : i + 1], expected, atol=0.05), (
            f"Device {i}: add mismatch"
        )

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(mgr)


# ── Test: ttnn.linear with origin-worker SubDevice (cols 0-5) ────────────────
def test_linear_origin_worker_sub_device(mesh_device):
    """
    ttnn.linear with compute_with_storage_grid_size=(6,1) on origin-worker
    SubDevice (cols 0-5). Grid starts from (0,0) so it stays within the
    worker SubDevice. This SHOULD pass.
    """
    worker_sd, sender_sd = make_origin_worker_sub_devices()
    mgr = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 3200)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    M, K, N = 32, 256, 192

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    a_tt = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    b_tt = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(6, 1),
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // 32,
        per_core_N=N // 32 // 6,
        fuse_batch=True,
        mcast_in0=True,
    )

    c_tt = ttnn.linear(
        a_tt, b_tt,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c_torch = ttnn.to_torch(c_tt, mesh_composer=mesh_composer)

    num_devices = mesh_device.get_num_devices()
    for i in range(num_devices):
        pcc = torch.nn.functional.cosine_similarity(
            c_torch[i : i + 1].flatten().float(),
            expected.flatten().float(),
            dim=0,
        ).item()
        assert pcc > 0.99, f"Device {i}: linear PCC {pcc:.4f} < 0.99"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(mgr)


# ── Test: matmul auto-grid with origin-worker SubDevice ──────────────────────
def test_matmul_auto_grid_origin_worker(mesh_device):
    """
    ttnn.matmul without explicit program_config, origin-worker SubDevice.
    Auto-config should pick a grid within cols 0-5.
    """
    worker_sd, sender_sd = make_origin_worker_sub_devices()
    mgr = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 3200)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    M, K, N = 32, 128, 128

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    a_tt = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    b_tt = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    c_tt = ttnn.matmul(a_tt, b_tt)
    c_torch = ttnn.to_torch(c_tt, mesh_composer=mesh_composer)

    num_devices = mesh_device.get_num_devices()
    for i in range(num_devices):
        pcc = torch.nn.functional.cosine_similarity(
            c_torch[i : i + 1].flatten().float(),
            expected.flatten().float(),
            dim=0,
        ).item()
        assert pcc > 0.99, f"Device {i}: matmul PCC {pcc:.4f} < 0.99"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(mgr)


# ── Test: EXPECTED FAILURE — matmul on realistic layout (cols 1-6) ───────────
@pytest.mark.xfail(
    reason="Matmul grid starts from (0,0) which spans sender SubDevice col 0. "
           "This is the core problem SubDevice migration must solve.",
    raises=RuntimeError,
    strict=True,
)
def test_matmul_realistic_layout_fails(mesh_device):
    """
    Documents the fundamental problem: matmul with compute_with_storage_grid_size
    always starts from (0,0). With realistic SubDevice layout (worker=cols 1-6,
    sender=cols 0,7), the matmul grid includes col 0 (sender) and the dispatch
    layer rejects it with "Programs must be executed on a single sub-device".

    This test is expected to FAIL (xfail). When we fix the matmul to support
    non-origin grids, this xfail should be removed.
    """
    worker_sd, sender_sd = make_realistic_sub_devices()
    mgr = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 3200)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    M, K, N = 32, 128, 128

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    a_tt = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    b_tt = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # This will raise RuntimeError: "Programs must be executed on a single sub-device"
    c_tt = ttnn.matmul(a_tt, b_tt)

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(mgr)
