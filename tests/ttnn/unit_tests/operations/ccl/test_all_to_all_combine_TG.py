# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import (
    run_all_to_all_combine_test,
    trace_all_to_all_combine,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D},
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((4, 8), (4, 8), id="4x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [1, 2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_all_to_all_combine_no_trace(
    mesh_device,
    trace_mode,
    mesh_shape,
    axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq,
    local_reduce,
    num_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if seq == 2 and ttnn.L1_MEMORY_CONFIG in (input_memory_config, output_memory_config):
        pytest.skip("Prefill needs to run in DRAM")

    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_combine_test(
        mesh_device,
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
        scheme="sequential",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
    )


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
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((4, 8), (4, 8), id="4x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(1, 40, 10), (128, 10, 5)],
    ids=["decode", "prefill"],
)
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_perf(
    mesh_device,
    mesh_shape,
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
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    trace_all_to_all_combine(
        mesh_device,
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
        "random",  # scheme TODO worst_perf
        dtype,
        topology,
        input_memory_config,
        output_memory_config,
    )
