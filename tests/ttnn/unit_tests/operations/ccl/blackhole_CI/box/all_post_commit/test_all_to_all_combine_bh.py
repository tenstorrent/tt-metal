# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import ttnn
from tracy import signpost
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)
from tests.nightly.t3000.ccl.test_all_to_all_combine import (
    run_all_to_all_combine_test,
)


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(3)
# fmt: off
@pytest.mark.parametrize(
    "device_params, num_devices, mesh_shape, axis, num_links, test_skew, topology, batches_per_device, experts_per_device, seq, select_experts_k, hidden_size, local_reduce, scheme, num_iters, input_memory_config, output_memory_config, dtype",
    [
        ({"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 500000}, 4, (4, 1), 0, 1, False, ttnn.Topology.Ring, 8, 8, 2, 8, 7000, True, "random", 2, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16),
        ({"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 500000}, 4, (4, 1), 0, 2, False, ttnn.Topology.Linear, 1, 256//4, 4096, 8, 7168, True, "random", 2, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
    ],
    ids=["decode", "prefill"], indirect=["device_params"])
# fmt: on
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
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

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
