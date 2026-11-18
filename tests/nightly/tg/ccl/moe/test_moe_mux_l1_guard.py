# SPDX-License-Identifier: Apache-2.0
"""Fabric-mux L1 map guard of the fused moe_compute combine (b46). The mux map is raw L1 outside the allocator, fitted at
the first program build under the lowest occupied L1 address. A later drop of the persistent L1 bottom (new persistent
tensor / semaphore) used to overrun the map on a program-cache hit (hang, RESULT_B39). With TT_MOE_MUX_L1_FIT_Q set the
program is keyed on the L1 bottom (rebuilt when it moves down by a quantum) and fitted Q below the build snapshot.

  case "grow":   call, allocate a persistent L1 tensor of GROW_KB per bank, call again (same shapes). Needs TT_MOE_MUX_L1_FIT_Q
                 (without it this is the B39 hang): asserts the second call rebuilt the program and still passes PCC.
  case "plain":  two calls, no growth: PCC passes; with the guard on, no extra program is built.
"""
import os

import pytest
import ttnn
from loguru import logger

from ttnn.operations.ccl import MoEActivationFunction, Topology
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import _run_moe_compute_impl

GUARD = int(os.environ.get("TT_MOE_MUX_L1_FIT_Q", "0"))
CHECK = int(os.environ.get("TT_MOE_MUX_L1_CHECK", "0"))
GROW_KB = int(os.environ.get("B46_GROW_KB", "160"))
COLS = int(os.environ.get("B46_COLS", "8"))
DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "trace_region_size": 750000,
    "l1_small_size": 16384,
}


def _call(mesh_device, mesh_shape, iters=1):
    hidden_size = 2048
    _run_moe_compute_impl(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=1,
        experts_per_device=4,
        tokens_per_device=32,
        selected_experts_k=8,
        num_layers=1,
        num_iterations=iters,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
        dtype=ttnn.bfloat16,
        enable_trace=False,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        topology=Topology.Linear,
        num_links=1,
        mux_core_range=((1, 1), (3, 3)),
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, COLS), (1, COLS))], indirect=["mesh_device"])
@pytest.mark.parametrize("case", ["plain", "grow"])
def test_mux_l1_guard(mesh_device, mesh_shape, case):
    if case == "grow" and not GUARD and CHECK < 2:
        pytest.skip(
            "grow without TT_MOE_MUX_L1_FIT_Q is the B39 hang; only run with TT_MOE_MUX_L1_CHECK=2 (raises instead of hanging)"
        )
    _call(mesh_device, mesh_shape, iters=2)
    n1 = mesh_device.num_program_cache_entries()
    keep = None
    if case == "grow":
        nb = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).num_banks
        keep = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, nb, GROW_KB * 1024 // 2]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.L1_MEMORY_CONFIG,
        )
    if case == "grow" and not GUARD:
        # stock behaviour: the program is reused with the mux map overlapping live L1. The debug check turns the B39 hang
        # into a deterministic error on the program-cache hit (before anything is enqueued).
        try:
            _call(mesh_device, mesh_shape, iters=2)
        except RuntimeError as e:
            assert "overlaps live L1" in str(e), str(e)[:500]
            logger.info(
                "B46 case=grow guard_q=0: stock reuse detected the mux-map overlap (would hang): " + str(e)[:300]
            )
            ttnn.deallocate(keep)
            return
        raise AssertionError("stock program reuse after L1 growth did not trip the mux-map overlap check")
    _call(mesh_device, mesh_shape, iters=2)
    n2 = mesh_device.num_program_cache_entries()
    logger.info(f"B46 case={case} guard_q={GUARD} cache entries {n1} -> {n2}")
    if GUARD and case == "grow":
        assert n2 > n1, "program was not rebuilt after the L1 bottom moved down"
    if case == "plain":
        # the call in _run_moe_compute_impl allocates other tensors, but moe_compute's own program must be reused
        logger.info("plain: PCC passed twice")
    if keep is not None:
        ttnn.deallocate(keep)
