# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run with TT_METAL_WATCHER=10 to check absent endpoint connection assertions."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1048576}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_shape, dtype",
    [
        pytest.param((1, 1, 1, 704), ttnn.bfloat16, id="bf16_single_channel"),
        pytest.param((1, 2, 1, 704), ttnn.bfloat8_b, id="bf8_paired_channels"),
    ],
)
def test_all_gather_async_linear_one_worker_endpoint(mesh_device, input_shape, dtype):
    """Both linear endpoints must handle their absent outward fabric connection."""
    mesh_device.enable_program_cache()
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    semaphores = [[ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)] for _ in range(2)]
    barriers = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]

    num_elements = input_shape[1] * input_shape[-1]
    local_pattern = (torch.arange(num_elements).reshape(input_shape) % 16).to(torch.bfloat16)
    host_parts = [local_pattern + 16 * rank for rank in range(4)]
    value = ttnn.from_torch(
        torch.cat(host_parts, dim=-1),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    # A copy collective must preserve the uploaded representation exactly,
    # independently of any conversion performed while uploading BF8 inputs.
    uploaded = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(value)]
    expected = torch.cat(uploaded, dim=-1)
    output_shape = (*input_shape[:-1], input_shape[-1] * 4)
    output = ttnn.empty(
        output_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def gather(ping_pong_index):
        return ttnn.experimental.all_gather_async(
            value,
            dim=3,
            cluster_axis=1,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=semaphores[ping_pong_index],
            barrier_semaphore=barriers[ping_pong_index],
            persistent_output_tensor=output,
            num_links=1,
            # One worker disables the mux; each outward endpoint still runs
            # its reader/writer pair but has no connection in that direction.
            num_workers_per_link=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def check_all_ranks(result):
        parts = ttnn.get_device_tensors(result)
        assert len(parts) == 4
        for rank, part in enumerate(parts):
            torch.testing.assert_close(
                ttnn.to_torch(part).float(), expected, rtol=0, atol=0, msg=f"Incorrect gather output on rank {rank}"
            )

    for ping_pong_index in range(2):
        result = gather(ping_pong_index)
        ttnn.synchronize_device(mesh_device)
        check_all_ranks(result)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        traced_result = gather(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        check_all_ranks(traced_result)
        for _ in range(8):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            check_all_ranks(traced_result)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
