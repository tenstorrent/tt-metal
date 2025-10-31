# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)
from tests.nightly.t3000.ccl.test_all_to_all_combine import (
    run_all_to_all_combine_test,
    trace_all_to_all_combine,
)


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_devices,cluster_axis, mesh_shape", [[8, 1, (1, 8)]])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(1, 40, 10)],
)
@pytest.mark.parametrize("local_reduce", [True])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_trace(
    bh_2d_mesh_device,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    local_reduce,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    batch = batches_per_device * num_devices
    experts = experts_per_device * num_devices
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    trace_all_to_all_combine(
        submesh_device,
        mesh_shape,
        cluster_axis,
        batch,
        seq_len,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        warmup_iters,
        num_links,
        "random",
        dtype,
        topology,
        input_memory_config,
        output_memory_config,
    )


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "device_params, num_links, test_skew,topology",
    [
        # FABRIC_1D LINE
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500000,
            },
            1,
            False,
            ttnn.Topology.Linear,
            id="fabric_1d_line_axis_0",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("num_devices, mesh_shape,axis", [(4, (4, 1), 0), (8, (1, 8), 1)])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_no_trace(
    bh_2d_mesh_device,
    mesh_shape,
    num_devices,
    axis,
    batches_per_device,
    seq,
    local_reduce,
    experts_per_device,
    select_experts_k,
    hidden_size,
    num_iters,
    scheme,
    input_memory_config,
    output_memory_config,
    num_links,
    topology,
    dtype,
    test_skew,
):
    if topology == ttnn.Topology.Ring:
        pytest.skip("Galaxy is currently mesh only")
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    bh_2d_mesh_device.disable_and_clear_program_cache()
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    run_all_to_all_combine_test(
        submesh_device,
        mesh_shape,
        axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme=scheme,
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        test_skew=test_skew,
    )
