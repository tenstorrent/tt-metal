# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import ttnn

# tests/ttnn/unit_tests/operations/cl/test_all_to_all_dispatch_t3000.py
from tests.ttnn.unit_tests.operations.ccl.test_all_to_all_combine_t3000 import (
    run_all_to_all_combine_test,
)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
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
    num_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_combine_test(
        mesh_device,
        mesh_shape,
        axis,
        batch,
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
