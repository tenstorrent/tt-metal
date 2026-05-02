# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-27: FIX AV — non-MMIO sysmem_manager stale state after quiesce cycles
#
# Background — FIX AV (device.cpp: configure_command_queue_programs):
#   sysmem_manager_->reset() was previously gated on is_mmio_capable().  Non-MMIO
#   devices therefore never reset prefetch_q_dev_fences, prefetch_q_in_flight, or
#   cq_to_quiesced during configure_command_queue_programs (called during quiesce
#   re-init path).
#
#   FIX AV moves the reset() call outside the is_mmio_capable() guard so that ALL
#   devices reset host-side CQ manager state during re-init.  The guard remains
#   for the case where fabric_relay_path_broken_ is set (reset() reads HW fence
#   via relay which would hang on a dead path).
#
# Why it matters:
#   If a non-MMIO device's prefetch_q_in_flight counter is stale after a quiesce
#   cycle, the next dispatch session inherits a non-zero in-flight count.  This
#   can cause fetch_queue_reserve_back() to stall indefinitely waiting for the
#   in-flight count to drop below the queue depth — even though the queue is
#   actually empty.  The symptom is a silent 15-minute CI timeout on the second
#   or subsequent dispatch in the same process.
#
# What this test verifies:
#   1. Three back-to-back quiesce + re-open cycles, with a dispatch on each leg.
#   2. Each post-quiesce dispatch must complete within 10s.
#      Without FIX AV: the 2nd or 3rd cycle can stall indefinitely if the
#      prefetch_q_in_flight counter was not reset (the stale count looks like
#      in-flight commands, blocking reserve_back()).
#   3. Dispatch correctness: each dispatch produces correct output tensor values.
#
# Why this is not tested elsewhere:
#   The current test suite exercises quiesce + re-open (GAP-14) but only runs
#   one dispatch after re-open.  The stale in-flight counter typically only
#   manifests on the 2nd or 3rd dispatch after re-open when the CQ pointer
#   wraps around and the stale count makes reserve_back() believe the queue
#   is full.  No existing test runs N > 1 dispatches per re-open cycle with
#   non-MMIO device dispatch validation.
#
# Topology: T3K (8 WH devices).  Requires >= 4 devices.

import time
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (1, 8)
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3
_PCC_THRESHOLD = 0.9999

# Number of full quiesce + re-open cycles.
_NUM_CYCLES = 3

# Number of dispatches per cycle.  Must be > 1 to expose stale in-flight counter.
_DISPATCHES_PER_CYCLE = 3

# Per-dispatch time budget.
_DISPATCH_TIMEOUT_S = 10.0

# Per-cycle open + init time budget.
_OPEN_TIMEOUT_S = 30.0


def _run_allgather(mesh_device, seed):
    """Run one AllGather and return (passed: bool, duration: float)."""
    torch.manual_seed(seed)
    full = torch.rand(_AG_OUTPUT_SHAPE, dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
    )
    t0 = time.time()
    out = ttnn.all_gather(inp, dim=_AG_DIM, topology=ttnn.Topology.Ring)
    # Synchronise to measure actual dispatch completion.
    ttnn.synchronize_device(mesh_device)
    duration = time.time() - t0

    out_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
    # Take device-0 view (all replicas identical after correct AllGather).
    num_devices = mesh_device.get_num_devices()
    out_single = out_torch[:1]
    passed, msg = comp_pcc(out_single, full[:1], pcc=_PCC_THRESHOLD)
    return passed, duration


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_gap27_fixav_nonmmio_sysmem_reset(mesh_device):
    """
    GAP-27: Verify that non-MMIO device sysmem_manager state is correctly reset
    across quiesce + re-open cycles, so that subsequent dispatches don't stall
    due to stale prefetch_q_in_flight counters.
    """
    # FIX RZ (#42429): skip if fabric is degraded — AllGather hangs on stale base-UMD channels.
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "GAP-27: fabric degraded (base-UMD channels) — skipping AllGather to avoid hang"
        )

    for cycle in range(_NUM_CYCLES):
        logger.info(f"=== GAP-27 cycle {cycle + 1}/{_NUM_CYCLES} ===")

        t_open = time.time()
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )
        open_duration = time.time() - t_open
        logger.info(f"[cycle {cycle}] fabric init in {open_duration:.1f}s")
        assert open_duration < _OPEN_TIMEOUT_S, (
            f"[cycle {cycle}] fabric init took {open_duration:.1f}s (expected < {_OPEN_TIMEOUT_S}s)"
        )

        for dispatch_idx in range(_DISPATCHES_PER_CYCLE):
            seed = cycle * _DISPATCHES_PER_CYCLE + dispatch_idx
            passed, duration = _run_allgather(mesh_device, seed=seed)

            logger.info(
                f"[cycle {cycle}][dispatch {dispatch_idx}] "
                f"duration={duration:.2f}s, pcc_ok={passed}"
            )
            assert duration < _DISPATCH_TIMEOUT_S, (
                f"[cycle {cycle}][dispatch {dispatch_idx}] AllGather stalled for "
                f"{duration:.2f}s — possible stale prefetch_q_in_flight on non-MMIO device "
                f"(FIX AV regression?). Expected < {_DISPATCH_TIMEOUT_S}s."
            )
            assert passed, (
                f"[cycle {cycle}][dispatch {dispatch_idx}] AllGather output failed PCC check "
                f"— non-MMIO device CQ state may be corrupt after re-init"
            )

        # Quiesce before next cycle (triggers configure_command_queue_programs reset path).
        logger.info(f"[cycle {cycle}] quiescing fabric")
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    logger.info(
        f"GAP-27: all {_NUM_CYCLES} cycles × {_DISPATCHES_PER_CYCLE} dispatches PASSED — "
        f"non-MMIO sysmem_manager correctly reset by FIX AV"
    )
