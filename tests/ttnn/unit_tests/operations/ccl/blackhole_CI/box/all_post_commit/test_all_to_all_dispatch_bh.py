# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

from tests.nightly.t3000.ccl.test_all_to_all_dispatch import (
    run_all_to_all_dispatch_test,
)


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(1)
# fmt: off
@pytest.mark.parametrize("device_params, trace_mode, num_devices, mesh_shape, cluster_axis, batches_per_device, experts_per_device, seq_len, num_iters, warmup_iters, select_experts_k, hidden_size, num_links, dtype, input_memory_config, output_memory_config",
    [
        ({"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, False, 4, (4, 1), 0, 8, 8, 128, 2, 1, 4, 7168, "MAX_LINKS", ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ({"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, False, 4, (4, 1), 0, 1, 256//4, 4096, 2, 1, 8, 7168, 1, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ({"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, False, 4, (4, 1), 0, 1, 256//4, 4096, 2, 1, 8, 7168, 2, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
    ],
    ids=["decode", "prefill_1", "prefill_2"], indirect=["device_params"])
# fmt: on
def test_all_to_all_dispatch_no_trace(
    bh_1d_mesh_device,
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
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, cluster_axis)
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]
    validate_test(dispatch_devices, topology, bh_1d_mesh_device.shape, 0)
    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = 1
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
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
@pytest.mark.parametrize("num_devices, mesh_shape", [(4, (4, 1))])
@pytest.mark.parametrize("cluster_axis", [0])
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
    bh_1d_mesh_device,
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
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, cluster_axis)
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = 1
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
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
