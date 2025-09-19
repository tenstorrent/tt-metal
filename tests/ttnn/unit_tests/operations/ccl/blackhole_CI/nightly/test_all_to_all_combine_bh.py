# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import sleep

import torch
import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
import ttnn
from tracy import signpost
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.nightly.test_all_gather_nightly import validate_test
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import (
    run_all_to_all_combine_test,
    trace_all_to_all_combine,
)

from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_dispatch_t3000 import (
    gen_tokens,
    gen_expert_mapping,
    get_metadata_tensor,
    get_expert_indices,
    get_output_tensor as get_sparse_tokens,
)


@skip_for_wormhole_b0("This test is for blackhole")
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
@pytest.mark.parametrize("num_devices", [4])
@pytest.mark.parametrize("mesh_shape", [(4, 1)])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [(1, 40, 10), (128, 10, 5)],
)
@pytest.mark.parametrize("local_reduce", [True])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_trace(
    bh_1d_mesh_device,
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
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, cluster_axis)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    trace_all_to_all_combine(
        bh_1d_mesh_device,
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
@pytest.mark.parametrize(
    "device_params, mesh_shape, num_devices, axis, num_links, test_skew",
    [
        # FABRIC_1D LINE
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500000,
            },
            (4, 1),
            4,
            0,
            1,
            False,
            id="fabric_1d_line_axis_0",
        ),
        # FABRIC_1D_RING
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500000,
            },
            (4, 1),
            4,
            0,
            1,
            False,
            id="fabric_1d_ring_axis_0",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_combine_no_trace(
    bh_1d_mesh_device,
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
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, axis)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    bh_1d_mesh_device.disable_and_clear_program_cache()

    run_all_to_all_combine_test(
        bh_1d_mesh_device,
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
