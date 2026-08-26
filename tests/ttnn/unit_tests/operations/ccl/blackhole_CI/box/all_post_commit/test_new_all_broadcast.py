# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import ttnn
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI._all_broadcast_helpers import (
    run_all_broadcast_impl as _run_all_broadcast_impl,
)

# Pre-refactor behavior of this file: CacheEntriesCounter attached to the mesh
# device with per-section measurement, and a fixed 1xN mesh-mapper placement.
run_all_broadcast_impl = partial(
    _run_all_broadcast_impl,
    cache_counter_mode="device_attr_sections",
    placement_mode="fixed_1xN",
)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, layout, input_dtype",
    [
        (2, 1, [1, 1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16),
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
    bh_1d_mesh_device,
    # pcie_mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    topology = ttnn.Topology.Linear
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, 0)
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    run_all_broadcast_impl(
        bh_1d_mesh_device,
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
    )
