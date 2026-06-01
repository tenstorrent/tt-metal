# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from models.demos.llama3_70b_galaxy.tt.model_config import PREFETCHER_NOC1_GRID

CLUSTER_SHAPE = (8, 4)
CLUSTER_AXIS = 1
NUM_LINKS = 2
M = 32
N = 1280
INPUT_NUM_CORES = 24
OUTPUT_NUM_CORES = 24

_DRAM_FATAL_RE = r"(?i)blackhole dram|does not support blackhole dram"

START_CORE = ttnn.CoreCoord(1, 0)


def _galaxy_sub_core_grids(max_y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, max_y)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, max_y)),
        ]
    )


def _ring_core_rangeset() -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in PREFETCHER_NOC1_GRID]
    )


def _round_up(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def _require_blackhole():
    if ttnn.get_arch_name().lower() != "blackhole":
        pytest.skip("Blackhole-only repro for all_reduce_async DRAM rejection")


def _setup_ccl_subdevice(mesh_device):
    sub_grids = _galaxy_sub_core_grids(7)
    worker_sub_device = ttnn.SubDevice([sub_grids])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    semaphores = [ttnn.create_global_semaphore(mesh_device, sub_grids, 0) for _ in range(2)]
    return sub_grids, worker_sub_device_id, semaphores


def _width_sharded_l1_memcfg(core_grid, m: int, n: int, num_cores: int) -> ttnn.MemoryConfig:
    n_per_shard = _round_up(math.ceil(n / num_cores), ttnn.TILE_SIZE)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [m, n_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _call_all_reduce_async(mesh_device, tt_input, worker_sub_device_id, semaphores):
    cluster_shape = CLUSTER_SHAPE
    output_crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        START_CORE, OUTPUT_NUM_CORES, _galaxy_sub_core_grids(7), row_wise=True
    )
    output_mem = _width_sharded_l1_memcfg(output_crs, M, N, OUTPUT_NUM_CORES)
    n_per_shard = _width_sharded_l1_memcfg(_ring_core_rangeset(), M, N, INPUT_NUM_CORES).shard_spec.shape[1]
    buffer_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_crs,
            [M, n_per_shard * cluster_shape[CLUSTER_AXIS]],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    tt_buffer = ttnn.from_torch(
        torch.zeros([*cluster_shape, M, n_per_shard * cluster_shape[CLUSTER_AXIS]]),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=buffer_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    ttnn.experimental.all_reduce_async(
        tt_input,
        tt_buffer,
        cluster_axis=CLUSTER_AXIS,
        mesh_device=mesh_device,
        multi_device_global_semaphore=semaphores[0],
        num_links=NUM_LINKS,
        memory_config=output_mem,
        dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
    )


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_all_reduce_async_rejects_dram_interleaved_on_blackhole(mesh_device):
    """DRAM interleaved input (typical `ttnn.linear(..., memory_config=None)` on BH)."""
    _require_blackhole()
    assert mesh_device.get_num_devices() == CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1]

    _, worker_sub_device_id, semaphores = _setup_ccl_subdevice(mesh_device)
    torch_input = torch.randn(1, 1, M, N)
    tt_dram = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    with pytest.raises(Exception, match=_DRAM_FATAL_RE):
        _call_all_reduce_async(mesh_device, tt_dram, worker_sub_device_id, semaphores)

    mesh_device.reset_sub_device_stall_group()
