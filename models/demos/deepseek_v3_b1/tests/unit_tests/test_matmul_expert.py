# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ExpertKernel (SRAM) and ExpertKernelDRAM (DRAM):
  SRAM: multi-device, multi-core matmul with compressed weights pre-loaded in L1.
  DRAM (single-device): B WIDTH_SHARDED in DRAM, streamed subblock by subblock.
  DRAM (multi-device): each device gets its own N-slice of B in DRAM.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import (
    ExpertKernel,
    ExpertKernelDRAM,
    create_dram_expert_tensors_multi_device,
    create_expert_fmt_tensors,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles
from models.demos.deepseek_v3_b1.tests.unit_tests.test_eltwise_add_compressed import scale_tiles_for_mixed_formats


def _scale_tiles_random_formats(b_torch, formats):
    """Randomly assign formats to tiles so the assigner picks a mix.

    Unlike scale_tiles_for_mixed_formats (which uses deterministic round-robin
    and can cause entire tile columns to share one format), this randomly assigns
    each tile a format. This avoids alignment issues with WIDTH_SHARDED cores.
    """
    if len(formats) <= 1:
        return

    M, N = b_torch.shape
    tiles_h, tiles_w = M // 32, N // 32

    for tr in range(tiles_h):
        for tc in range(tiles_w):
            fmt = formats[torch.randint(0, len(formats), (1,)).item()]
            r0, r1 = tr * 32, (tr + 1) * 32
            c0, c1 = tc * 32, (tc + 1) * 32
            if fmt == "bfp8":
                for r in range(32):
                    b_torch[r0 + r, c0:c1] *= 2.0 ** (r % 16)
            elif fmt == "bfp2":
                for r in range(32):
                    exp = torch.randint(-3, 4, (1,)).float()
                    signs = torch.sign(torch.randn(32))
                    signs[signs == 0] = 1.0
                    b_torch[r0 + r, c0:c1] = signs * (2.0**exp)
            elif fmt == "bfp0":
                b_torch[r0:r1, c0:c1] = torch.randn(32, 32) * 1e-3
            # bfp4: keep randn as-is


def _make_index_tensor(expert_idx: int, core_grid, mesh_device) -> ttnn.Tensor:
    """Create a HEIGHT_SHARDED uint16 index tensor, same format as DRAMStreamingExpertsMatmul.

    Each core on each device gets a [1, 16] shard with expert_idx in element 0.
    """
    num_cores = core_grid.num_cores()
    index_torch = torch.zeros(num_cores, 16, dtype=torch.int32)
    index_torch[:, 0] = expert_idx
    index_torch = index_torch.to(torch.uint16)
    index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [1, 16], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        index_torch,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=index_mem_config,
        tile=ttnn.Tile([1, 16]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _run_multi_device(mesh_device, M, K, N, formats_per_device, core_grid, pcc_threshold=0.98):
    """
    Each device gets B[:,i*N_dev:(i+1)*N_dev], independently compressed with its own formats.
    N is the per-core shard width; total N per device = N × num_cores.

    B is reshaped to [mesh_rows, mesh_cols, K, N_per_device] and distributed via
    PlacementShard(0)/PlacementShard(1) so device (r,c) gets full K but its own N slice.
    A is replicated across cores and devices. Output is sharded along N across all devices.
    """
    num_devices = mesh_device.get_num_devices()
    assert len(formats_per_device) == num_devices

    num_cores = core_grid.num_cores()
    mesh_rows = mesh_device.shape[0]
    mesh_cols = mesh_device.shape[1]
    N_per_device = N * num_cores
    N_total = N_per_device * num_devices
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)

    torch_b_slices = []
    for dev_idx, formats in enumerate(formats_per_device):
        torch.manual_seed(dev_idx * 42)
        b_slice = torch.randn((K, N_per_device)).float()
        _scale_tiles_random_formats(b_slice, formats)
        torch_b_slices.append(b_slice)

    # Stack per-device slices into [mesh_rows, mesh_cols, K, N_per_device].
    # PlacementShard(0)/PlacementShard(1) gives device (r,c) full K, unique N slice.
    torch_b_4d = torch.stack(torch_b_slices).reshape(mesh_rows, mesh_cols, K, N_per_device)

    all_formats = list({fmt for fmts in formats_per_device for fmt in fmts})
    bfp0_mae = 1e-3 if "bfp0" in all_formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=all_formats, bfp0_mae_threshold=bfp0_mae)

    b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [K, N], ttnn.ShardOrientation.ROW_MAJOR),
    )
    logger.info(f"Uploading expert tensor: shape={list(torch_b_4d.shape)}, {num_cores} core(s)")
    ct = CompressedTensor.from_torch(
        torch_b_4d,
        assigner,
        device=mesh_device,
        memory_config=b_mem_config,
        per_core_allocation=True,
        mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
    )
    logger.info("Expert tensor uploaded")

    a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    a_mesh = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=a_mem_config,
        tile=ttnn.Tile([M, 32]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [M, N], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mesh = ttnn.from_torch(
        torch.zeros((M, N_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=out_mem_config,
        tile=ttnn.Tile([M, 32]),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    num_tiles = (K // 32) * (N // 32)
    logger.info(
        f"Uploading expert fmt tensors: 1 expert, {num_tiles} tiles/core, {num_cores} core(s), {num_devices} device(s)"
    )
    fmt_tensors = create_expert_fmt_tensors([ct], mesh_device, core_grid, num_tiles)
    logger.info("Running ExpertKernel.op")
    index_tensor = _make_index_tensor(0, core_grid, mesh_device)
    result_mesh = ExpertKernel.op(a_mesh, [ct], out_mesh, fmt_tensors_per_device=fmt_tensors, index_tensor=index_tensor)

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result_mesh)):
        output_torch = ttnn.to_torch(out_dev)
        torch_expected = (torch_a.float() @ torch_b_slices[dev_idx]).bfloat16()
        passing, msg = comp_pcc(torch_expected, output_torch, pcc_threshold)
        logger.info(f"Device {dev_idx} ({formats_per_device[dev_idx]}) PCC: {msg}")
        assert passing, f"Device {dev_idx} ({formats_per_device[dev_idx]}) failed: {msg}"


def _run_multi_device_multi_expert(
    mesh_device, M, K, N, formats_per_device, core_grid, num_experts, selected_expert_idx, pcc_threshold=0.97
):
    """
    Multi-expert variant: each device holds num_experts separate CompressedTensors.
    N is the per-core shard width; total N per device = N × num_cores.

    Experts on the same device may have different formats. At runtime, selected_expert_idx
    picks which expert to compute, verified against the corresponding torch reference.

    The fmt table is [num_experts, num_tiles] packed entries per core; the kernel
    offsets by expert_idx * num_tiles to select the right row — no offset arithmetic
    needed since addresses are stored explicitly per expert.
    """
    num_devices = mesh_device.get_num_devices()
    assert len(formats_per_device) == num_devices

    num_cores = core_grid.num_cores()
    mesh_rows = mesh_device.shape[0]
    mesh_cols = mesh_device.shape[1]
    N_per_device = N * num_cores
    N_total = N_per_device * num_devices
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)

    # Build per-device, per-expert weight tensors and CompressedTensors.
    # Expert e on device d uses formats_per_device[d] (same formats for all experts on a device).
    torch_b_experts = []  # [num_experts][num_devices] = torch tensor (K, N_per_device)
    cts = []  # [num_experts] = CompressedTensor

    b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [K, N], ttnn.ShardOrientation.ROW_MAJOR),
    )

    all_formats = list({fmt for fmts in formats_per_device for fmt in fmts})
    print(f"All formats used across devices: {all_formats}")
    bfp0_mae = 1e-3 if "bfp0" in all_formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=all_formats, bfp0_mae_threshold=bfp0_mae)

    for expert_idx in range(num_experts):
        expert_slices = []
        for dev_idx, formats in enumerate(formats_per_device):
            torch.manual_seed(expert_idx * 1000 + dev_idx * 42)
            b_slice = torch.randn((K, N_per_device)).float()
            _scale_tiles_random_formats(b_slice, all_formats)
            expert_slices.append(b_slice)
        torch_b_experts.append(expert_slices)

        torch_b_4d = torch.stack(expert_slices).reshape(mesh_rows, mesh_cols, K, N_per_device)
        logger.info(f"Uploading expert {expert_idx}/{num_experts}: shape={list(torch_b_4d.shape)}, {num_cores} core(s)")
        ct = CompressedTensor.from_torch(
            torch_b_4d,
            assigner,
            device=mesh_device,
            memory_config=b_mem_config,
            per_core_allocation=True,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        logger.info(f"Expert {expert_idx} uploaded")
        cts.append(ct)

    a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    a_mesh = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=a_mem_config,
        tile=ttnn.Tile([M, 32]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [M, N], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mesh = ttnn.from_torch(
        torch.zeros((M, N_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=out_mem_config,
        tile=ttnn.Tile([M, 32]),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    num_tiles = (K // 32) * (N // 32)
    logger.info(
        f"Uploading expert fmt tensors: {num_experts} expert(s), {num_tiles} tiles/core, {num_cores} core(s), {num_devices} device(s)"
    )
    fmt_tensors = create_expert_fmt_tensors(cts, mesh_device, core_grid, num_tiles)
    logger.info(f"Running ExpertKernel.op with expert_idx={selected_expert_idx}")
    index_tensor = _make_index_tensor(selected_expert_idx, core_grid, mesh_device)
    result_mesh = ExpertKernel.op(a_mesh, cts, out_mesh, fmt_tensors_per_device=fmt_tensors, index_tensor=index_tensor)

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result_mesh)):
        output_torch = ttnn.to_torch(out_dev)
        torch_expected = (torch_a.float() @ torch_b_experts[selected_expert_idx][dev_idx]).bfloat16()
        passing, msg = comp_pcc(torch_expected, output_torch, pcc_threshold)
        logger.info(f"Device {dev_idx} expert {selected_expert_idx} ({formats_per_device[dev_idx]}) PCC: {msg}")
        assert passing, f"Device {dev_idx} expert {selected_expert_idx} failed: {msg}"


# ---------------------------------------------------------------------------
# Multi-device, multi-core tests (4×2 mesh, 8 devices)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
@pytest.mark.parametrize(
    "N, core_grid",
    [
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                }
            ),
            id="N32_4cores",
        ),
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                }
            ),
            id="N32_6cores",
        ),
        pytest.param(
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            id="N128_9cores",
        ),
    ],
)
def test_expert_kernel_multi_device_same_formats(bh_2d_mesh_device, N, core_grid):
    """All 8 devices use bfp8+bfp4, B sharded along N across cores and devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_multi_device(mesh, M=1, K=7168, N=N, formats_per_device=[["bfp8", "bfp4"]] * 8, core_grid=core_grid)


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
@pytest.mark.parametrize(
    "N, core_grid",
    [
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                }
            ),
            id="N32_4cores",
        ),
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                }
            ),
            id="N32_6cores",
        ),
        pytest.param(
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            id="N128_9cores",
        ),
    ],
)
def test_expert_kernel_multi_device_different_formats(bh_2d_mesh_device, N, core_grid):
    """Each device gets its own compressed format: alternating bfp8+bfp4 and bfp8+bfp2."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    formats = [["bfp8", "bfp4"], ["bfp8", "bfp2"]] * 4
    _run_multi_device(mesh, M=1, K=7168, N=N, formats_per_device=formats, core_grid=core_grid)


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
@pytest.mark.parametrize(
    "N, core_grid",
    [
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                }
            ),
            id="N32_4cores",
        ),
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                }
            ),
            id="N32_6cores",
        ),
        pytest.param(
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            id="N128_9cores",
        ),
    ],
)
def test_expert_kernel_multi_device_asymmetric_formats(bh_2d_mesh_device, N, core_grid):
    """Mixed: some devices bfp8 only, others all four formats."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    formats = [["bfp8"], ["bfp8", "bfp4"], ["bfp8", "bfp2"], ["bfp8", "bfp4", "bfp2", "bfp0"]] * 2
    _run_multi_device(mesh, M=1, K=7168, N=N, formats_per_device=formats, core_grid=core_grid)


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
@pytest.mark.parametrize(
    "N, core_grid",
    [
        pytest.param(
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                }
            ),
            id="N32_8cores",
        ),
        pytest.param(
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            id="N128_9cores",
        ),
    ],
)
def test_expert_kernel_multi_device_wide(bh_2d_mesh_device, N, core_grid):
    """Wider N output, alternating formats across 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    formats = [["bfp8", "bfp4"], ["bfp8", "bfp2"]] * 4
    _run_multi_device(mesh, M=1, K=7168, N=N, formats_per_device=formats, core_grid=core_grid)


# ---------------------------------------------------------------------------
# Multi-expert, multi-core tests (4×2 mesh, 8 experts per device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
@pytest.mark.parametrize("selected_expert_idx", [4])
@pytest.mark.parametrize(
    "N, core_grid",
    [
        pytest.param(
            64,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            id="N64_9cores",
        ),
    ],
)
def test_expert_kernel_multi_expert(bh_2d_mesh_device, selected_expert_idx, N, core_grid):
    """8 experts per device, same formats, verify first/middle/last expert selection."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_multi_device_multi_expert(
        mesh,
        M=1,
        K=7168,
        N=N,
        formats_per_device=[["bfp4", "bfp0"]] * 8,
        core_grid=core_grid,
        num_experts=8,
        selected_expert_idx=selected_expert_idx,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
@pytest.mark.parametrize("selected_expert_idx", [3])
@pytest.mark.parametrize(
    "N, core_grid",
    [
        pytest.param(
            64,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            id="N64_9cores",
        ),
    ],
)
def test_expert_kernel_multi_expert_mixed_formats(bh_2d_mesh_device, selected_expert_idx, N, core_grid):
    """8 experts per device with mixed formats per device, verify selection correctness."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    formats = [
        ["bfp4", "bfp2"],
        ["bfp4"],
        ["bfp4", "bfp2", "bfp0"],
        ["bfp4", "bfp0"],
        ["bfp2", "bfp0"],
        ["bfp4", "bfp2"],
        ["bfp4"],
        ["bfp2"],
    ]
    _run_multi_device_multi_expert(
        mesh,
        M=1,
        K=7168,
        N=N,
        formats_per_device=formats,
        core_grid=core_grid,
        num_experts=8,
        selected_expert_idx=selected_expert_idx,
    )


# ---------------------------------------------------------------------------
# DRAM expert path helpers and tests
# ---------------------------------------------------------------------------


def _pad_to_dram_banks(n, tile_w, lcm):
    remainder = n % lcm
    return n if remainder == 0 else n + (lcm - remainder)


def _run_dram_expert_multi_device(
    mesh_device,
    M,
    K,
    N,
    formats_per_device,
    num_experts,
    selected_expert_idx,
    subblock_k=None,
    cores_per_bank=1,
    pcc_threshold=0.97,
):
    """
    Multi-device DRAM expert: each device streams its own N-slice from DRAM.
    B is [K, N_per_device] per device, distributed via PlacementShard.
    A is replicated, output is sharded along N across devices.
    """
    tile_w = 32
    num_devices = mesh_device.get_num_devices()
    mesh_rows = mesh_device.shape[0]
    mesh_cols = mesh_device.shape[1]
    assert len(formats_per_device) == num_devices

    dev0 = ttnn.get_device_tensors(
        ttnn.from_torch(
            torch.zeros(num_devices, 1, dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
    )[0].device()
    primary_cores_list = dev0.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(primary_cores_list)
    num_cores = num_banks * cores_per_bank

    n_padded = _pad_to_dram_banks(N, tile_w, tile_w * num_banks * cores_per_bank)
    per_core_N = n_padded // (num_banks * cores_per_bank)  # elements per compute core
    N_per_device = n_padded
    N_total = N_per_device * num_devices

    Kt = K // tile_w
    if subblock_k is None:
        subblock_k = Kt // 4 if Kt > 8 else Kt
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)
    assert Kt % subblock_k == 0
    num_subblocks_k = Kt // subblock_k

    compute_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(c.x + offset, c.y), ttnn.CoreCoord(c.x + offset, c.y))
            for c in primary_cores_list
            for offset in range(cores_per_bank)
        ]
    )

    logger.info(
        f"DRAM expert multi-device: M={M}, K={K}, N_per_device={N_per_device}, "
        f"num_experts={num_experts}, selected={selected_expert_idx}, "
        f"num_devices={num_devices}, num_banks={num_banks}"
    )

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)

    all_formats = list({fmt for fmts in formats_per_device for fmt in fmts})
    bfp0_mae = 1e-3 if "bfp0" in all_formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=all_formats, bfp0_mae_threshold=bfp0_mae)

    # DRAM memory config (same for all devices — PlacementShard distributes per-device data)
    dram_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dev0.dram_grid_size().x - 1, dev0.dram_grid_size().y - 1))]
    )
    total_N_per_bank = per_core_N * cores_per_bank
    b_shard_spec = ttnn.ShardSpec(dram_grid, [K, total_N_per_bank], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, b_shard_spec)

    logger.info(f"Creating {num_experts} expert CTs in DRAM ({num_devices} devices)...")
    torch_b_experts = {}
    cts = []
    for eidx in range(num_experts):
        expert_slices = []
        for dev_idx in range(num_devices):
            torch.manual_seed(eidx * 1000 + dev_idx * 137)
            b_raw = torch.randn((K, N_per_device)).float()
            scale_tiles_for_mixed_formats(b_raw, formats_per_device[dev_idx])
            expert_slices.append(b_raw)
            coord = ttnn.MeshCoordinate(dev_idx // mesh_cols, dev_idx % mesh_cols)
            if coord not in torch_b_experts:
                torch_b_experts[coord] = []
            torch_b_experts[coord].append(b_raw)

        # Shuffle each device's slice for column-major DRAM layout
        slices_shuffled = [shuffle_tensor_tiles(s, tile_w, num_banks) for s in expert_slices]
        torch_b_4d = torch.stack(slices_shuffled).reshape(mesh_rows, mesh_cols, K, N_per_device)

        ct = CompressedTensor.from_torch(
            torch_b_4d,
            assigner,
            device=mesh_device,
            memory_config=b_mem_config,
            per_core_allocation=False,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        cts.append(ct)
        logger.info(f"  CT {eidx}/{num_experts} uploaded")

    logger.info("Creating A, output, index mesh tensors...")
    # A: HEIGHT_SHARDED, replicated across devices
    a_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    a_mesh = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=a_mem_config,
        tile=ttnn.Tile([M, tile_w]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Output: WIDTH_SHARDED per device, ShardTensorToMesh across devices
    out_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    out_mesh = ttnn.from_torch(
        torch.zeros((M, N_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=out_mem_config,
        tile=ttnn.Tile([M, tile_w]),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    # Index: HEIGHT_SHARDED, replicated
    index_torch = torch.zeros(num_cores, 16, dtype=torch.int32)
    index_torch[:, 0] = selected_expert_idx
    index_torch = index_torch.to(torch.uint16)
    index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, [1, 16], ttnn.ShardOrientation.ROW_MAJOR),
    )
    index_mesh = ttnn.from_torch(
        index_torch,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=index_mem_config,
        tile=ttnn.Tile([1, 16]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Creating DRAM expert device tensors (in1_backing + metadata)...")
    device_data = create_dram_expert_tensors_multi_device(
        mesh_device, cts, subblock_k, num_subblocks_k, per_core_N // tile_w, cores_per_bank=cores_per_bank
    )

    logger.info("Running ExpertKernelDRAM.op...")
    result_mesh = ExpertKernelDRAM.op(a_mesh, cts, out_mesh, device_data, index_mesh, subblock_k)

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result_mesh)):
        output_torch = ttnn.to_torch(out_dev)
        coord = ttnn.MeshCoordinate(dev_idx // mesh_cols, dev_idx % mesh_cols)
        torch_expected = (torch_a.float() @ torch_b_experts[coord][selected_expert_idx].float()).bfloat16()
        passing, msg = comp_pcc(torch_expected, output_torch, pcc_threshold)
        logger.info(f"Device {dev_idx} expert {selected_expert_idx} PCC: {msg}")
        assert passing, f"Device {dev_idx} failed: {msg}"


# ---------------------------------------------------------------------------
# DRAM expert tests — all use _run_dram_expert_multi_device (single device = 1×1 mesh)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_single_bfp8(device):
    """Single expert, bfp8 only — baseline sanity check."""
    _run_dram_expert_multi_device(
        device, M=1, K=128, N=256, formats_per_device=[["bfp8"]], num_experts=1, selected_expert_idx=0, subblock_k=2
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_multi_expert_select_first(device):
    """4 experts, select expert 0."""
    _run_dram_expert_multi_device(
        device, M=1, K=7168, N=2048, formats_per_device=[["bfp8", "bfp4"]], num_experts=4, selected_expert_idx=0
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_multi_expert_select_last(device):
    """4 experts, select expert 3."""
    _run_dram_expert_multi_device(
        device, M=1, K=7168, N=2048, formats_per_device=[["bfp8", "bfp4"]], num_experts=4, selected_expert_idx=3
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_multi_expert_mixed_formats(device):
    """4 experts, mixed bfp4+bfp2+bfp0."""
    _run_dram_expert_multi_device(
        device, M=1, K=7168, N=2048, formats_per_device=[["bfp4", "bfp2", "bfp0"]], num_experts=4, selected_expert_idx=2
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_8experts_select_middle(device):
    """8 experts, bfp8+bfp4, select expert 5."""
    _run_dram_expert_multi_device(
        device, M=1, K=7168, N=2048, formats_per_device=[["bfp8", "bfp4"]], num_experts=8, selected_expert_idx=5
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_2cores_per_bank(device):
    """2 cores per bank, 4 experts."""
    _run_dram_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=2048,
        formats_per_device=[["bfp4", "bfp2"]],
        num_experts=4,
        selected_expert_idx=1,
        cores_per_bank=2,
    )


# ---------------------------------------------------------------------------
# DRAM expert multi-device tests (4×2 mesh, 8 devices)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_multi_device_same_formats(bh_2d_mesh_device):
    """8 devices, same bfp8+bfp4 formats, 4 experts, select expert 2."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_dram_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=2048,
        formats_per_device=[["bfp8", "bfp4"]] * 8,
        num_experts=4,
        selected_expert_idx=2,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"allocator_mode": ttnn.per_core_allocation.AllocatorMode.HYBRID}],
    indirect=True,
)
def test_dram_expert_multi_device_mixed_formats(bh_2d_mesh_device):
    """8 devices, alternating formats, 4 experts, select expert 3."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    formats = [
        ["bfp4", "bfp2"],
        ["bfp4"],
        ["bfp4", "bfp2", "bfp0"],
        ["bfp4", "bfp0"],
        ["bfp2", "bfp0"],
        ["bfp4", "bfp2"],
        ["bfp4"],
        ["bfp2"],
    ]
    _run_dram_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=2048,
        formats_per_device=formats,
        num_experts=4,
        selected_expert_idx=3,
    )
