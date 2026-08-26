# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)

# The shared helper defaults (no cache-entries counter, fixed 8x4 mesh-mapper
# placement) match this file's pre-refactor behavior.
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI._all_broadcast_helpers import run_all_broadcast_impl


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, layout, input_dtype, cluster_axis",
    [
        (4, 1, [1, 1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16, 1),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_broadcast(
    bh_2d_mesh_device,
    # pcie_mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    cluster_axis,
):
    topology = ttnn.Topology.Linear
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    run_all_broadcast_impl(
        bh_2d_mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_broadcast_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        cluster_axis=cluster_axis,
    )
