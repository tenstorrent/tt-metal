# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_to_all_combine import (
    run_all_to_all_combine_test,
    trace_all_to_all_combine,
)


# Standard 8x4 TG tests - comprehensive fabric and memory config coverage
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        },
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        },
        # {
        #    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        #    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        #    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        #    "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        # },# test removed due to issue 35320
        # {
        #    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        #    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        #    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        #    "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        # },# test removed due to issue 35320
    ],
    ids=[
        "fabric_2d",
        "fabric_1d_line",
        "fabric_1d_ring",
        # "fabric_manager_enabled_2d",# test removed due to issue 35320
        # "fabric_manager_enabled_1d_line",# test removed due to issue 35320
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 4), (8, 4), id="8x4_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2], ids=["s2"])
@pytest.mark.parametrize("local_reduce", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("num_links", [4], ids=["num_links_4"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
@pytest.mark.parametrize(
    "input_memory_config, output_memory_config, axis",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, 0),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 1),
    ],
    ids=["dram_in_l1_out_axis0", "l1_in_dram_out_axis1"],
)
def test_all_to_all_combine_8x4(
    mesh_device,
    trace_mode,
    mesh_shape,
    axis,
    batches_per_device,
    experts,
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
    batch = batches_per_device * mesh_shape[axis]

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


# Dual galaxy (8x8) tests - only the 2 tests run in CI (matching dispatch pattern)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
    ],
    ids=["fabric_1d_line"],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 8), (8, 8), id="8x8_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("axis", [1], ids=["axis_1"])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [2], ids=["s2"])
@pytest.mark.parametrize("local_reduce", [True], ids=["sparse"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("num_links", [1], ids=["num_links_1"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_all_to_all_combine_8x8_dual_galaxy(
    mesh_device,
    trace_mode,
    mesh_shape,
    axis,
    batches_per_device,
    experts,
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
    batch = batches_per_device * mesh_shape[axis]

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


# Quad galaxy (8x16) tests - single test for basic validation
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
    ],
    ids=["fabric_1d_line"],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 16), (8, 16), id="8x16_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("axis", [1], ids=["axis_1"])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("seq", [1], ids=["s1"])
@pytest.mark.parametrize("local_reduce", [True], ids=["sparse"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("num_links", [1], ids=["num_links_1"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_all_to_all_combine_8x16_quad_galaxy(
    mesh_device,
    trace_mode,
    mesh_shape,
    axis,
    batches_per_device,
    experts,
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
    batch = batches_per_device * mesh_shape[axis]

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


# Performance test - kept as is
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(1, 40, 10), (128, 10, 5)],
    ids=["decode", "prefill"],
)
@pytest.mark.parametrize("local_reduce", [True])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_perf(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts,
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
