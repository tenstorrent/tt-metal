# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole


def run_deepseek_moe_reduce_scatter_impl(
    mesh_device,
    num_devices,
    dtype,
    layout,
    pre_rs_reduction_dim,
    pre_rs_reduction_input_shape,
    sum_input_memory_config,
    rs_input_memory_config,
    rs_output_memory_config,
    rs_dim,
    rs_num_links=1,
    rs_topology=None,
    rs_cluster_axis=None,
    enable_trace=True,
    num_iters=1,
):
    torch.manual_seed(0)

    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        pre_rs_reduction_global_input_shape = pre_rs_reduction_input_shape[:]
        pre_rs_reduction_global_input_shape[rs_dim] *= num_devices
        pre_rs_reduction_input_tensor = torch.rand(pre_rs_reduction_global_input_shape).bfloat16()
        input_tensors = torch.chunk(pre_rs_reduction_input_tensor, num_devices, rs_dim)
        torch_input_tensor_list.append(input_tensors)

        input_tensor_mesh = ttnn.from_torch(
            pre_rs_reduction_input_tensor,
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=sum_input_memory_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(rs_dim)], ttnn.MeshShape(1, num_devices)
                ),
            ),
        )

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        pre_rs_reduction_output = [torch.sum(t, dim=0, keepdim=True) for t in torch_input_tensor_list[i]]
        reduce_output = torch.sum(torch.stack(pre_rs_reduction_output), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, rs_dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        tt_pre_rs_reduction_output = ttnn.experimental.deepseek_moe_fast_reduce_nc(
            tt_input_tensor_mesh_list[i],
            dim=pre_rs_reduction_dim,
            split_size=int(pre_rs_reduction_input_shape[-1] / num_devices),
            output_memory_config=rs_input_memory_config,
        )

        tt_reduce_scatter_output_tensor = ttnn.experimental.deepseek_moe_reduce_scatter(
            tt_pre_rs_reduction_output,
            output_memory_config=rs_output_memory_config,
            dim=rs_dim,
            num_links=rs_num_links,
            topology=rs_topology,
            cluster_axis=rs_cluster_axis,
        )

        return tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        tt_reduce_scatter_output_trace_list = []
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_trace_list.append(tt_reduce_scatter_output_tensor)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        for tt_tensor in tt_reduce_scatter_output_trace_list:
            tt_rs_out = ttnn.from_device(tt_tensor)
            tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=rs_dim))
            tt_tensor.deallocate(True)
            tt_reduce_scatter_output_list.append(tt_rs_out)
    else:
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_rs_out = ttnn.from_device(tt_reduce_scatter_output_tensor)
            tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=rs_dim))
            tt_reduce_scatter_output_tensor.deallocate(True)
            tt_reduce_scatter_output_list.append(tt_rs_out)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_rs_out = tt_reduce_scatter_output_list[i]
        torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

        torch_rs_out = torch.cat(torch_rs_out_tensor, rs_dim)
        eq, output = comp_pcc(tt_rs_out, torch_rs_out)

        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED: {output}"

    mesh_device.reset_sub_device_stall_group()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT)])
@pytest.mark.parametrize("pre_rs_reduction_dim", [(0)])
@pytest.mark.parametrize(
    "pre_rs_reduction_input_shape, sum_input_memory_config, rs_input_memory_config, rs_output_memory_config, rs_dim, rs_num_links",
    [
        (
            [8, 1, 32, 2048],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            1,
        ),  # one_link
        (
            [8, 1, 32, 1024],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            1,
        ),  # one_link_partial
    ],
    ids=[
        "one_link",
        "one_link_partial",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize("enable_trace, num_iters", [(True, 10)])
def test_deepseek_moe_reduce_scatter(
    mesh_device,
    dtype,
    layout,
    pre_rs_reduction_dim,
    pre_rs_reduction_input_shape,
    sum_input_memory_config,
    rs_input_memory_config,
    rs_output_memory_config,
    rs_dim,
    rs_num_links,
    rs_topology,
    enable_trace,
    num_iters,
):
    run_deepseek_moe_reduce_scatter_impl(
        mesh_device=mesh_device,
        num_devices=mesh_device.get_num_devices(),
        dtype=dtype,
        layout=layout,
        pre_rs_reduction_dim=pre_rs_reduction_dim,
        pre_rs_reduction_input_shape=pre_rs_reduction_input_shape,
        sum_input_memory_config=sum_input_memory_config,
        rs_input_memory_config=rs_input_memory_config,
        rs_output_memory_config=rs_output_memory_config,
        rs_dim=rs_dim,
        rs_num_links=rs_num_links,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )
