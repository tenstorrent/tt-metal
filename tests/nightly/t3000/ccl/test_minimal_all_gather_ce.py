# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parity coverage for ``ttnn.experimental.all_gather_ce`` using the same harness as ``test_all_gather_async``.

The implementation under ``all_gather_ce`` shares host routing with ``all_gather_async`` (including composite
all-gather). The minimal fabric path uses a separate device op and kernel sources under
``all_gather_ce/device/kernels/``. We do not duplicate ``test_all_gather_async_broadcast`` here because
``all_gather_ce`` rejects the via-broadcast program variant (use ``all_gather_async`` for that).
"""

import pytest
import ttnn

from models.common.utility_functions import skip_for_blackhole

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters, use_barrier, use_persistent_buffers, pcc_threshold",
    [
        (
            [1, 1, 1024, 5120],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            True,
            True,
            1.0,
        ),  # perf, barrier_with_persistent
        (
            [8, 1, 512, 512],
            0,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            True,
            False,
            1.0,
        ),  # check, barrier_without_persistent
        (
            [1, 1, 1024, 1024],
            2,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            False,
            True,
            1.0,
        ),  # perf, no_barrier_with_persistent
        (
            [1, 1, 1024, 1024],
            -1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            False,
            True,
            1.0,
        ),  # perf, no_barrier_with_persistent
        (
            [1, 1, 48, 1024],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            True,
            True,
            1.0,
        ),  # check, barrier_with_persistent
        (
            [1, 1, 48, 1024],
            -1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            True,
            True,
            1.0,
        ),  # check, barrier_with_persistent
        # Composite-AG tests
        (
            [1, 1, 1, 8],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            True,
            False,
            1.0,
        ),  # perf, barrier_without_persistent
        (
            [1, 16, 32, 32],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            False,
            True,
            1.0,
        ),  # check, no_barrier_with_persistent
        (
            [1, 1, 1024, 5120],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            False,
            1,
            True,
            True,
            0.9999,
        ),  # perf, barrier_with_persistent
    ],
    ids=[
        "sd35_spatial-perf-barrier_with_persistent",
        "gather_dim_0-check-barrier_without_persistent",
        "gather_dim_2-perf-no_barrier_with_persistent",
        "gather_dim_negative_2-perf-no_barrier_with_persistent",
        "gather_dim_3_padded_dim_2-check-barrier_with_persistent",
        "gather_dim_negative_1_padded_dim_2-check-barrier_with_persistent",
        "composite_ag_test_two-perf-barrier_without_persistent",
        "composite_ag_test_four-check-no_barrier_with_persistent",
        "sd35_spatial-perf-barrier_with_persistent_bfloat8_b",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_ce(
    mesh_device,
    num_links,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    use_barrier,
    use_persistent_buffers,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    pcc_threshold,
):
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        use_semaphore_free_all_gather_impl=False,
        allowed_pcc=pcc_threshold,
        all_gather_function=ttnn.experimental.all_gather_ce,
    )
