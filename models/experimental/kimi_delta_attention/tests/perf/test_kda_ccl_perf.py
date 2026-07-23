# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eight-device output all-reduce roofline for target-shape KDA prefill."""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

NUM_DEVICES = 8
NUM_LINKS = 2
TOKENS = 640
HIDDEN_SIZE = 2304
LINK_GBPS_PER_DIRECTION = 400.0

pytestmark = [
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.skipif(os.environ.get("CI") == "true", reason="local perf test"),
    pytest.mark.parametrize("mesh_device", [(1, NUM_DEVICES)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def test_kda_output_all_reduce_perf(mesh_device: ttnn.MeshDevice) -> None:
    """Measure the replicated-output collective proposed for TP=8 KDA."""
    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("roofline assumptions apply to Blackhole")
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("real-time profiler must be active for KDA CCL perf")

    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker = ttnn.SubDevice([cores])
    worker_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([worker], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_id])

    rs_semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]
    ag_semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
    barrier_semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
    host_input = torch.ones(NUM_DEVICES, 1, TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        host_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)],
                ttnn.MeshShape(1, NUM_DEVICES),
            ),
        ),
    )

    def run() -> ttnn.Tensor:
        return ttnn.experimental.all_reduce_async(
            input_tensor,
            num_devices=NUM_DEVICES,
            barrier_semaphores=barrier_semaphores,
            rs_global_semaphores=rs_semaphores,
            ag_global_semaphores=ag_semaphores,
            math_op=ttnn.ReduceType.Sum,
            num_links=NUM_LINKS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_id,
        )

    warm_output = run()
    ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_id])
    ttnn.deallocate(warm_output)
    output, records = profile_realtime_program(mesh_device, run, collect_all=True)
    ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_id])

    durations_by_program: dict[int, float] = {}
    for record in records:
        runtime_id = int(record["runtime_id"])
        if runtime_id:
            durations_by_program[runtime_id] = max(
                durations_by_program.get(runtime_id, 0.0),
                float(record["duration_ns"]),
            )
    measured_ns = sum(durations_by_program.values())
    assert measured_ns > 0

    local_bytes = TOKENS * HIDDEN_SIZE * torch.bfloat16.itemsize
    critical_path_bytes = 2 * local_bytes * (NUM_DEVICES - 1) / NUM_DEVICES
    roofline_gbytes_per_second = LINK_GBPS_PER_DIRECTION * NUM_LINKS / 8
    theoretical_ns = critical_path_bytes / roofline_gbytes_per_second
    logger.info(
        f"KDA TP=8 all-reduce: payload={local_bytes / 1e6:.3f} MB, "
        f"critical-path={critical_path_bytes / 1e6:.3f} MB, "
        f"roofline={roofline_gbytes_per_second:.1f} GB/s, "
        f"theoretical={theoretical_ns / 1e3:.3f} us, measured={measured_ns / 1e3:.3f} us, "
        f"utilization={theoretical_ns / measured_ns:.1%}"
    )
    assert math.isclose(float(ttnn.to_torch(ttnn.get_device_tensors(output)[0]).mean()), NUM_DEVICES, abs_tol=0.01)

    ttnn.deallocate(output)
    ttnn.deallocate(input_tensor)
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
