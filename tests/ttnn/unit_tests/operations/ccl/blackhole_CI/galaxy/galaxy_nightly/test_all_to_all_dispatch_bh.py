# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)


from tests.nightly.t3000.ccl.test_all_to_all_dispatch import (
    run_all_to_all_dispatch_test,
)


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize("num_devices,mesh_shape,cluster_axis", [(4, (4, 1), 0), (8, (1, 8), 1)])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batches_per_device, seq_len, num_iters, warmup_iters",
    [
        (16, 2, 2, 1),
    ],
    ids=["b16s2"],
)
@pytest.mark.parametrize("num_links", ["MAX_LINKS"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
def test_all_to_all_dispatch_no_trace(
    bh_2d_mesh_device,
    trace_mode,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    topology = ttnn.Topology.Linear
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = 1
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    run_all_to_all_dispatch_test(
        submesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
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
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize("num_devices, mesh_shape, cluster_axis", [(8, (1, 8), 1)])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [4])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (128, 2, 1),
    ],
    ids=["s128"],
)
@pytest.mark.parametrize(
    "input_memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
    ],
    ids=["dram"],
)
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", ["MAX_LINKS"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_dispatch_trace(
    bh_2d_mesh_device,
    trace_mode,
    mesh_shape,
    num_devices,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = 1
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    run_all_to_all_dispatch_test(
        submesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )
