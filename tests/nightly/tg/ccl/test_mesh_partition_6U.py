# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

# Import the test function from the t3000 file
from tests.nightly.t3000.ccl.test_mesh_partition import (
    run_mesh_partition_test,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 22000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("per_device_output_shape, dim", [((1, 1, 8, 7168), 2)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("cluster_axis", [0, 1, None])
@pytest.mark.parametrize("mesh_axes", [[0, 1]])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_mesh_partition_rm(
    mesh_device,
    mesh_shape,
    trace_mode,
    per_device_output_shape,
    dtype,
    layout,
    dim,
    cluster_axis,
    mesh_axes,
    input_memory_config,
    output_memory_config,
):
    num_iters = 2
    warmup_iters = 0

    run_mesh_partition_test(
        mesh_device,
        per_device_output_shape,
        dim,
        num_iters,
        warmup_iters,
        trace_mode,
        dtype,
        layout,
        cluster_axis,
        mesh_axes,
        mesh_shape,
        input_memory_config,
        output_memory_config,
        scheme="random",
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1048576}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mesh_partition_cached_slice_args(mesh_device):
    """Retain coordinate-specific tile arguments across fresh buffers and cache/trace reuse."""
    import torch

    cases = [
        (ttnn.TILE_LAYOUT, 2, 0, [1, 4, 64, 128]),
        (ttnn.TILE_LAYOUT, 2, 1, [1, 4, 64, 128]),
        (ttnn.TILE_LAYOUT, 2, None, [1, 4, 64, 128]),
        (ttnn.TILE_LAYOUT, 3, 1, [1, 4, 64, 128]),
        (ttnn.TILE_LAYOUT, 1, 0, [1, 4, 64, 128]),
        (ttnn.ROW_MAJOR_LAYOUT, 2, 0, [1, 4, 64, 128]),
        # Three/six output tiles or rows exercise a partial active-core run and idle slots.
        (ttnn.TILE_LAYOUT, 2, 0, [1, 1, 32, 96]),
        (ttnn.ROW_MAJOR_LAYOUT, 2, 0, [1, 1, 3, 128]),
        # Partition starts differ by 130 bytes, exercising the RM reader's W-offset fixup.
        (ttnn.ROW_MAJOR_LAYOUT, 3, 1, [1, 1, 3, 65]),
    ]
    for case_index, (layout, dim, axis, base_shape) in enumerate(cases):
        group_size = mesh_device.get_num_devices() if axis is None else mesh_device.shape[axis]
        retained_tensors = []
        for repeat in range(3):
            shape = list(base_shape)
            shape[dim] *= group_size
            shape[2] *= 1 + repeat % 2
            torch.manual_seed(case_index * 10 + repeat)
            host_input = torch.rand(shape, dtype=torch.bfloat16)
            device_input = ttnn.from_torch(
                host_input,
                dtype=ttnn.bfloat16,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            expected_shards = torch.chunk(host_input, group_size, dim=dim)

            def standalone_slice():
                ends = list(shape)
                ends[dim] //= group_size
                result = ttnn.slice(device_input, starts=[0] * len(shape), ends=ends, steps=[1] * len(shape))
                for local in ttnn.get_device_tensors(result):
                    assert torch.equal(ttnn.to_torch(local), expected_shards[0])
                retained_tensors.append(result)

            # These kernels are shared with Slice, whose unique-address ABI must remain valid
            # both before and after the MeshPartition-only common-address variant is cached.
            standalone_slice()
            entries_before = mesh_device.num_program_cache_entries()
            output = ttnn.mesh_partition(device_input, dim=dim, cluster_axis=axis)
            ttnn.synchronize_device(mesh_device)
            if repeat == 2:
                assert mesh_device.num_program_cache_entries() == entries_before
            # Keep previous allocations alive so a cache hit must patch genuinely different addresses.
            retained_tensors.extend((device_input, output))
            expected_shards = torch.chunk(host_input, group_size, dim=dim)

            def check(actual_output):
                for rank, local_output in enumerate(ttnn.get_device_tensors(actual_output)):
                    partition = rank if axis is None else rank // 4 if axis == 0 else rank % 4
                    assert torch.equal(ttnn.to_torch(local_output), expected_shards[partition])

            check(output)
            standalone_slice()
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = ttnn.mesh_partition(device_input, dim=dim, cluster_axis=axis)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            check(traced_output)
        finally:
            ttnn.release_trace(mesh_device, trace_id)

        # Re-enter eager dispatch after trace retargeting, queue distinct fresh buffers, and
        # keep all allocations alive so allocator reuse cannot hide a stale address.
        entries_after_trace = mesh_device.num_program_cache_entries()
        queued = []
        for offset in (1, 2):
            next_host = host_input + offset
            next_input = ttnn.from_torch(
                next_host,
                dtype=ttnn.bfloat16,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            next_output = ttnn.mesh_partition(next_input, dim=dim, cluster_axis=axis)
            retained_tensors.extend((next_input, next_output))
            queued.append((next_output, torch.chunk(next_host, group_size, dim=dim)))
        ttnn.synchronize_device(mesh_device)
        assert mesh_device.num_program_cache_entries() == entries_after_trace
        for actual_output, expected_shards in queued:
            check(actual_output)
