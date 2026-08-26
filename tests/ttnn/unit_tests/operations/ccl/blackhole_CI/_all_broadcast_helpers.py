# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the test_new_all_broadcast.py test files under
box/all_post_commit, box/nightly, galaxy/galaxy_nightly and galaxy/nightly.

The four copies of these helpers had drifted apart in three ways; each
divergence is preserved behind an optional parameter so every call site keeps
its exact pre-refactor behavior:

- ``cache_counter_mode`` selects how program-cache entries are counted and
  asserted:
    - ``None`` (galaxy copies): no CacheEntriesCounter; the final assertion
      reads ``mesh_device.num_program_cache_entries()`` directly.
    - ``"device_attr_sections"`` (box/all_post_commit): a CacheEntriesCounter
      is attached to ``mesh_device.cache_entries_counter``; the compile run and
      trace capture (inside run_with_trace) and the non-trace dispatch loop are
      each measured separately.
    - ``"local_wrap_dispatch"`` (box/nightly): a local CacheEntriesCounter
      wraps the whole dispatch section (trace or eager) in a single measure().
- ``placement_mode`` selects the mesh-mapper placement:
    - ``"fixed_8x4"`` (galaxy copies): shard onto a fixed MeshShape(8, 4).
    - ``"fixed_1xN"`` (box/all_post_commit): shard onto MeshShape(1, num_devices).
    - ``"detect"`` (box/nightly): derive row/column orientation from
      ``mesh_device.shape``.
- ``cluster_axis`` is forwarded to ``ttnn.all_broadcast`` for every mode. The
  box/all_post_commit copy used to omit the kwarg entirely (and never receives
  a non-None value from its tail); passing ``cluster_axis=None`` explicitly is
  equivalent — the box/nightly copy already does exactly that on the same
  hardware.
"""

import contextlib

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.tests_common.cache_entries_counter import CacheEntriesCounter


def run_with_trace(
    mesh_device,
    all_broadcast_topology,
    input_tensor_mesh,
    num_links,
    output_mem_config,
    num_iter=20,
    subdevice_id=None,
    cluster_axis=None,
    cache_counter=None,
):
    measure = cache_counter.measure if cache_counter is not None else contextlib.nullcontext

    # Compile Run
    logger.info("Compiling model")
    with measure():
        tt_out_tensor = ttnn.all_broadcast(
            input_tensor_mesh,
            num_links=num_links,
            cluster_axis=cluster_axis,
            memory_config=output_mem_config,
            topology=all_broadcast_topology,
            subdevice_id=subdevice_id,
        )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with measure():
        for i in range(num_iter):
            tt_out_tensor = ttnn.all_broadcast(
                input_tensor_mesh,
                num_links=num_links,
                cluster_axis=cluster_axis,
                memory_config=output_mem_config,
                topology=all_broadcast_topology,
                subdevice_id=subdevice_id,
            )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


def run_all_broadcast_impl(
    mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    function_level_defaults,
    all_broadcast_topology,
    num_iters=1,
    trace_mode=False,
    rand_tensor=True,
    mem_config=None,
    input_shard_shape=None,
    input_shard_grid=None,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
    cluster_axis=None,
    cache_counter_mode=None,
    placement_mode="fixed_8x4",
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    if cache_counter_mode == "device_attr_sections":
        mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"input_shard_shape: {input_shard_shape}")
    logger.info(f"input_shard_grid: {input_shard_grid}")

    ### For sharded all broadcast only
    if bool(input_shard_shape) != bool(input_shard_grid) and bool(tensor_mem_layout) != bool(input_shard_grid):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )
    if input_shard_shape and input_shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        if output_shard_shape is None:
            assert (
                output_shard_grid is None
            ), "output_shard_grid must not be provided if output_shard_shape is not provided"
            output_shard_shape = input_shard_shape
            output_shard_spec = ttnn.ShardSpec(
                input_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
        else:
            assert output_shard_grid is not None, "output_shard_grid must be provided if output_shard_shape is provided"
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
    else:
        assert mem_config is not None
        input_mem_config = mem_config
        output_mem_config = mem_config
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters):
        output_tensors = []
        for k in range(num_devices):
            if rand_tensor:
                output_tensor = torch.rand(output_shape).bfloat16()
            else:
                output_tensor = torch.zeros(output_shape)
                row_id = 1
                # Fix indices for tensors with ranks 2 or 3
                for w in range(output_shape[0]):
                    for z in range(output_shape[1]):
                        for y in range(0, output_shape[2], 32):
                            for x in range(0, output_shape[3], 32):
                                output_tensor[w, z, y, :] = row_id
                                row_id += 1
            output_tensors.append(output_tensor)

        output_tensor_goldens_list.append(output_tensors)
        temp_output_tensor = torch.cat(output_tensors, -1)

        if placement_mode == "detect":
            # Detect actual mesh shape and configure accordingly
            mesh_actual_shape = mesh_device.shape
            if mesh_actual_shape[0] > 1 and mesh_actual_shape[1] == 1:
                # Row-oriented: (N, 1)
                placement = [ttnn.PlacementShard(-1), ttnn.PlacementReplicate()]
                logical_shape = ttnn.MeshShape(num_devices, 1)
            elif mesh_actual_shape[1] > 1 and mesh_actual_shape[0] == 1:
                # Column-oriented: (1, N)
                placement = [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
                logical_shape = ttnn.MeshShape(1, num_devices)
            else:
                # Default to column-oriented for other cases
                placement = [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
                logical_shape = ttnn.MeshShape(1, num_devices)
        elif placement_mode == "fixed_1xN":
            placement = [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
            logical_shape = ttnn.MeshShape(1, num_devices)
        else:
            assert placement_mode == "fixed_8x4", f"unknown placement_mode: {placement_mode}"
            placement = [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
            logical_shape = ttnn.MeshShape(8, 4)

        input_tensor_mesh = ttnn.from_torch(
            temp_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(placement, logical_shape),
            ),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if cache_counter_mode == "local_wrap_dispatch":
        cache_entries_counter = CacheEntriesCounter(mesh_device)
        with cache_entries_counter.measure():
            if trace_mode:
                tt_out_tensor = run_with_trace(
                    mesh_device,
                    all_broadcast_topology,
                    input_tensor_mesh_list[0],
                    num_links,
                    output_mem_config,
                    num_iter=num_iters,
                    subdevice_id=worker_sub_device_id,
                    cluster_axis=cluster_axis,
                )
                tt_out_tensor_list.append(tt_out_tensor)
            else:
                for i in range(num_iters):
                    tt_out_tensors = ttnn.all_broadcast(
                        input_tensor_mesh_list[i],
                        num_links=num_links,
                        cluster_axis=cluster_axis,
                        memory_config=output_mem_config,
                        topology=all_broadcast_topology,
                        subdevice_id=worker_sub_device_id,
                    )
                    tt_out_tensor_list.append(tt_out_tensors)

                logger.info(f"Waiting for op")
                ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
                logger.info(f"Done op")
    elif cache_counter_mode == "device_attr_sections":
        if trace_mode:
            tt_out_tensor = run_with_trace(
                mesh_device,
                all_broadcast_topology,
                input_tensor_mesh_list[0],
                num_links,
                output_mem_config,
                num_iter=num_iters,
                subdevice_id=worker_sub_device_id,
                cache_counter=mesh_device.cache_entries_counter,
            )
            tt_out_tensor_list.append(tt_out_tensor)
        else:
            with mesh_device.cache_entries_counter.measure():
                for i in range(num_iters):
                    tt_out_tensors = ttnn.all_broadcast(
                        input_tensor_mesh_list[i],
                        num_links=num_links,
                        memory_config=output_mem_config,
                        topology=all_broadcast_topology,
                        subdevice_id=worker_sub_device_id,
                    )
                    tt_out_tensor_list.append(tt_out_tensors)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")
    else:
        assert cache_counter_mode is None, f"unknown cache_counter_mode: {cache_counter_mode}"
        if trace_mode:
            tt_out_tensor = run_with_trace(
                mesh_device,
                all_broadcast_topology,
                input_tensor_mesh_list[0],
                num_links,
                output_mem_config,
                num_iter=num_iters,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )
            tt_out_tensor_list.append(tt_out_tensor)
        else:
            for i in range(num_iters):
                tt_out_tensors = ttnn.all_broadcast(
                    input_tensor_mesh_list[i],
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=all_broadcast_topology,
                    subdevice_id=worker_sub_device_id,
                    cluster_axis=cluster_axis,
                )
                tt_out_tensor_list.append(tt_out_tensors)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_out_tensors = tt_out_tensor_list[tensor_index]
        output_tensors = output_tensor_goldens_list[tensor_index]
        for k in range(num_devices):
            output_tensor = output_tensors[k]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensors[k])):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                logger.info(f"Checking for device {t.device().id()}")
                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, output_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                    passed = False
                    assert eq, f"{i} FAILED: {output}"
    if cache_counter_mode == "local_wrap_dispatch":
        assert (
            cache_entries_counter.total == 1 or cache_entries_counter.total == num_iters
        ), f"Device has {cache_entries_counter.total} program cache entries"
    elif cache_counter_mode == "device_attr_sections":
        assert (
            mesh_device.cache_entries_counter.total == 1 or mesh_device.cache_entries_counter.total == num_iters
        ), f"Device has {mesh_device.cache_entries_counter.total} program cache entries"
    else:
        assert (
            mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
        ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    if not passed:
        assert eq, f"{i} FAILED: {output}"
