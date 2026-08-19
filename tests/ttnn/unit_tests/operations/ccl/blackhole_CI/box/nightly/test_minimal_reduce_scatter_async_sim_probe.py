# SPDX - FileCopyrightText : © 2026 Tenstorrent USA, Inc.
#
# SPDX - License - Identifier : Apache - 2.0

"""
craq-sim verification probe for reduce_scatter_minimal_async's RING writer on 8 Blackhole chips.

WHY THIS EXISTS: the ring writer kernel is only reached with `Topology.Ring`, and Ring is only legal
when the collective spans the ENTIRE row so it loops around (`validate_test` rejects a 4-device ring
on an 8-device line). The shipped 8-device Ring reduce-scatter cases live in the t3000 test, which is
`@skip_for_blackhole`, while the Blackhole ring cases are all num_devices=4 — so nothing shipped
exercises the ring writer on the simulator's reliable 8-chip all-MMIO Blackhole configuration.

This probe reuses the Blackhole test's own `run_reduce_scatter_impl` with num_devices=8 so the ring
writer — migrated onto the CCL dataflow helper's MuxConn policy and armed scatter/unicast/inc/
multicast-inc channels — is actually executed and PCC-verified rather than only compiled.

MULTI-BATCH COVERAGE: the shapes below deliberately include one with a leading dim > 1, so
`input_tensor_B > 1` and the schedule's per-batch restart of the ring slice walk is executed. That
path matters: the ring slice cursor is re-seeded to the same first slice at the top of every batch,
and a version that instead let batch N+1 continue where batch N stopped reads and writes the wrong
slice on every step after the first batch — silently, with no hang and no CB mismatch. A B=1 shape
cannot distinguish the two. (The bug was caught by the host equivalence sweep over
ccl_helpers_schedule.hpp rather than here, which is the argument for having both.)

Verification vehicle, not a CI test: gated on the simulator so it skips on hardware.
"""

import os

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_minimal_reduce_scatter_async_bh import (
    run_reduce_scatter_impl,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("TT_METAL_SIMULATOR"),
    reason="craq-sim multichip verification probe; requires TT_METAL_SIMULATOR",
)


@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "rs_input_shape, dim",
    [
        # dim = 3 over a[1, 1, 32, 8 * 256] input scatters 256 columns to each of the 8 ring members.
        ([1, 1, 32, 2048], 3),
        # Same scatter, but input_tensor_B = 2 so the schedule's per-batch ring restart is exercised.
        ([2, 1, 32, 2048], 3),
        # dim = 0 selects a DIFFERENT kernel triple(dim_zero_ring_*), so it needs its own case.
        ([8, 1, 32, 256], 0),
    ],
    ids=["b1", "b2", "dim0"],
)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config_input", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("mem_config_rs", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("num_iters", [2])  # >1 so the program-cache-hit path is exercised too
# use_barrier = True drives the multicast - inc barrier channel; the batch - ready multicast inc runs either way.
@pytest.mark.parametrize("use_barrier", [True])
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456},
            ttnn.Topology.Ring,
        )
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async_ring_8dev_sim(
    bh_1d_mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    num_iters,
    use_barrier,
):
    run_reduce_scatter_impl(
        bh_1d_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology,
        num_iters=num_iters,
        enable_trace=False,  # trace replay is a separate concern from the dataflow migration
        use_barrier=use_barrier,
    )


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
# LINE topology.Ring and Line select entirely different kernel triples
# (line_reduce_scatter_minimal_async_{reader, writer } + line_reduction.cpp, or the dim_zero_line_ *
# variants when dim == 0), so covering Ring above says nothing about these.Same two dim cases.
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "rs_input_shape, dim",
    [
        ([1, 1, 32, 2048], 3),
        ([8, 1, 32, 256], 0),
    ],
    ids=["dim3", "dim0"],
)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config_input", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("mem_config_rs", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456},
            ttnn.Topology.Linear,
        )
    ],
    indirect=["device_params"],
    ids=["fabric_line"],
)
def test_reduce_scatter_async_line_8dev_sim(
    bh_1d_mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    num_iters,
):
    run_reduce_scatter_impl(
        bh_1d_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology,
        num_iters=num_iters,
        enable_trace=False,
    )
