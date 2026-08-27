# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI._all_to_all_helpers import run_all_to_all_impl


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize(
    "num_links, logical_shape, in_dim, out_dim, layout",
    [
        (1, [1, 1, 44544, 3072 * 3], 2, 3, ttnn.TILE_LAYOUT),  # Pre-attn all-to-all
    ],
    ids=["pre-attn"],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize(
    "num_iters, do_check, reuse_inputs",
    [(2, True, False)],
    ids=["check"],
)
@pytest.mark.parametrize(
    "enable_trace",
    [True, False],
    ids=["use_trace", "no_trace"],
)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_all_to_all(
    bh_1d_mesh_device,
    logical_shape,
    in_dim,
    out_dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    do_check,
    reuse_inputs,
    enable_trace,
    is_ci_env,
):
    num_devices = bh_1d_mesh_device.shape[0]
    topology = ttnn.Topology.Ring
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, 0)
    run_all_to_all_impl(
        bh_1d_mesh_device,
        num_devices,
        logical_shape,
        in_dim,
        out_dim,
        num_links,
        input_dtype,
        layout,
        topology=topology,
        num_iters=num_iters,
        mem_config=mem_config,
        do_check=do_check,
        trace_mode=enable_trace,
        reuse_inputs=reuse_inputs,
    )
