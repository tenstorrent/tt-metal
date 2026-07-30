# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Correctness coverage for ttnn.experimental.reduce_scatter_minimal_direct.

The direct (one-shot) reduce-scatter unicasts each destination's slice straight to that destination
instead of relaying it around the ring. It is Ring-only and op-managed (no caller semaphores, no
barrier semaphore), so the matrix here is deliberately narrower than the minimal_async one: small
shapes, two scatter dims, trace on and off, and the three persistent-buffer modes.

`run_reduce_scatter_minimal_direct_impl` is shared -- tests/nightly/tg/ccl and the blackhole_CI
box/galaxy modules import it and only supply their own device fixture and parametrization.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Persistent-buffer modes, spelled out because they select different code paths in the op:
#   "both"    -- {output, staging} from the op's helper. The writer's start barrier is COMPILE-TIME
#                skipped in this mode (a pinned staging address makes it unnecessary).
#   "staging"  -- caller-owned output + reduce_scatter_minimal_direct_create_staging_buffer, the
#                convenience path for callers that already have an output tensor.
#   "none"    -- op allocates both, which is what compiles the start barrier in.
PERSISTENT_MODES = ["both", "staging", "none"]


def run_reduce_scatter_minimal_direct_impl(
    mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    num_iters=1,
    enable_trace=True,
    cluster_axis=None,
    persistent_mode="both",
):
    """rs_input_shape is the PER-DEVICE input; the global tensor is that with `dim` scaled by
    num_devices and sharded across the mesh, so the op reduces over num_devices and scatters `dim`."""
    torch.manual_seed(0)
    assert persistent_mode in PERSISTENT_MODES, f"unknown persistent_mode {persistent_mode}"

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    rs_output_shape = rs_input_shape[:]
    rs_output_shape[dim] //= num_devices

    logger.info(f"reduce_scatter_minimal_direct shape {rs_input_shape} dim {dim} dtype {rs_input_dtype}")
    logger.info(f"num_devices {num_devices} num_links {num_links} cluster_axis {cluster_axis}")
    logger.info(f"trace {enable_trace} num_iters {num_iters} persistent {persistent_mode}")

    ##### Input setup: a distinct tensor per iteration, so a stale result cannot pass #####
    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []
    for _ in range(num_iters):
        rs_global_input_shape = rs_input_shape[:]
        rs_global_input_shape[dim] *= num_devices
        rs_input_tensor = torch.rand(rs_global_input_shape).bfloat16()
        torch_input_tensor_list.append(torch.chunk(rs_input_tensor, num_devices, dim))
        tt_input_tensor_mesh_list.append(
            ttnn.from_torch(
                rs_input_tensor,
                device=mesh_device,
                layout=layout,
                dtype=rs_input_dtype,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    ttnn.MeshMapperConfig(
                        [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
                    ),
                ),
            )
        )

    # The op reads its topology from the hardware rather than taking it as an argument, and hard-fails
    # on anything but a ring. Ask what this placement actually resolves to and skip if the devices along
    # cluster_axis do not wrap -- a partial row/column is a line, not a small ring.
    usable_topology = ttnn.get_usable_topology(tt_input_tensor_mesh_list[0], cluster_axis=cluster_axis)
    if usable_topology != ttnn.Topology.Ring:
        pytest.skip(
            f"reduce_scatter_minimal_direct is Ring-only; {num_devices} device(s) on cluster_axis "
            f"{cluster_axis} of mesh {tuple(mesh_device.shape)} resolve to {usable_topology}"
        )

    ##### Persistent buffers, one set per iteration (so every iteration's output is verifiable) #####
    persistent_buffers_list = []
    for i in range(num_iters):
        if persistent_mode == "both":
            persistent_buffers_list.append(
                ttnn.experimental.reduce_scatter_minimal_direct_create_persistent_buffers(
                    tt_input_tensor_mesh_list[i], dim=dim, cluster_axis=cluster_axis
                )
            )
        elif persistent_mode == "staging":
            output_buffer = ttnn.from_torch(
                torch.zeros(rs_output_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=rs_input_dtype,
                memory_config=mem_config_rs,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            staging_buffer = ttnn.experimental.reduce_scatter_minimal_direct_create_staging_buffer(
                tt_input_tensor_mesh_list[i], dim=dim, cluster_axis=cluster_axis
            )
            persistent_buffers_list.append([output_buffer, staging_buffer])
        else:
            persistent_buffers_list.append(None)

    ##### Torch golden #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        torch_reduce_scatter_output_list.append(torch.chunk(reduce_output, num_devices, dim))

    def run_op(i):
        return ttnn.experimental.reduce_scatter_minimal_direct(
            tt_input_tensor_mesh_list[i],
            dim=dim,
            cluster_axis=cluster_axis,
            num_links=num_links,
            memory_config=mem_config_rs,
            persistent_buffers=persistent_buffers_list[i],
            subdevice_id=worker_sub_device_id,
        )

    tt_reduce_scatter_output_list = []
    if enable_trace:
        for i in range(num_iters):  # compile
            run_op(i)
        logger.info("Done compiling op")

        trace_output_list = []
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            trace_output_list.append(run_op(i))
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info("Done capturing trace")

        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info("Done executing trace")

        for tt_tensor in trace_output_list:
            tt_rs_out = ttnn.from_device(tt_tensor)
            tt_reduce_scatter_output_list.append(
                ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
            )
        ttnn.release_trace(mesh_device, trace_id)
    else:
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_rs_out = ttnn.from_device(tt_reduce_scatter_output_tensor)
            tt_reduce_scatter_output_list.append(
                ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
            )
            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        torch_rs_out = torch.cat(torch_reduce_scatter_output_list[i], dim)
        eq, output = comp_pcc(tt_reduce_scatter_output_list[i], torch_rs_out)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"iteration {i} FAILED: {output}"

    mesh_device.reset_sub_device_stall_group()


# Small shapes, two scatter dims. Every scatter dim is 8 tiles/pages wide, so the same shapes are
# valid on any ring size that divides 8 (2, 4, 8) and the destinations below can size their ring
# from the mesh they are handed.
RS_DIRECT_SHAPES = [
    ([1, 1, 32, 256], 3),  # scatter the last dim: 8 tiles wide -> 1 tile/device at N=8
    ([1, 1, 256, 128], 2),  # scatter height: 8 tiles tall -> 1 tile/device at N=8
]
RS_DIRECT_SHAPE_IDS = ["dim3_w256", "dim2_h256"]

RS_DIRECT_TRACE_CASES = [(True, 3), (False, 2)]
RS_DIRECT_TRACE_IDS = ["trace", "no_trace"]

RS_DIRECT_DRAM_MEM_CONFIG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_SHAPES, ids=RS_DIRECT_SHAPE_IDS)
@pytest.mark.parametrize("enable_trace, num_iters", RS_DIRECT_TRACE_CASES, ids=RS_DIRECT_TRACE_IDS)
@pytest.mark.parametrize("persistent_mode", PERSISTENT_MODES)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct(
    mesh_device,
    num_links,
    rs_input_shape,
    dim,
    rs_input_dtype,
    enable_trace,
    num_iters,
    persistent_mode,
):
    run_reduce_scatter_minimal_direct_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        ttnn.TILE_LAYOUT,
        RS_DIRECT_DRAM_MEM_CONFIG,
        RS_DIRECT_DRAM_MEM_CONFIG,
        num_iters=num_iters,
        enable_trace=enable_trace,
        persistent_mode=persistent_mode,
    )
