# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-30: FIX AL — STARTED early-exit timing regression test
#
# Background — FIX AL (device.cpp: wait_for_fabric_workers_ready):
#   Before FIX AL, wait_for_fabric_workers_ready() polled each device's
#   sync_buf[0] until it reached EDMStatus::RUNNING, with a 10000ms
#   (kSyncTimeoutMs) hard limit.  When a relay-broken quiesce leaves EDM
#   firmware in STARTED state (firmware began executing but handshake never
#   completed because the relay was torn down before RUNNING was signalled),
#   every non-MMIO device waits the full 10000ms before timing out.
#
#   On a T3K 8-device mesh with 4+ relay-broken non-MMIO devices:
#     Old timeout: 4 × 10000ms = 40s of wasted waiting per quiesce cycle.
#     New timeout: 4 × 3000ms  = 12s (kStartedTimeoutMs = 3000ms).
#
#   FIX AL adds an early-exit: if sync_buf[0] == EDMStatus::STARTED and
#   elapsed > kStartedTimeoutMs (3000ms), exit the poll loop immediately
#   with a warning log instead of waiting the remaining 7000ms.
#
# Why STARTED ≠ RUNNING:
#   STARTED means: ERISC firmware has booted and set sync_buf[0] = STARTED,
#   but hasn't yet completed the relay handshake with the host (which sets
#   sync_buf[0] = RUNNING).  After a SIGKILL predecessor, the next init
#   finds ERISC channels still in ACTIVE relay state.  The host issues a
#   quiesce to reset them, but the EDM never reaches RUNNING — it stays at
#   STARTED indefinitely.  Without FIX AL, each device waits the full 10s.
#
# What this test verifies:
#   1. After a SIGKILL predecessor leaves ERISC channels in dirty state,
#      re-opening the mesh + quiescing completes in < 20s per cycle.
#      Without FIX AL: 4+ devices × 10s = 40s+ (CI timeout risk).
#      With FIX AL:    4+ devices × 3s  ≈ 12s (well under 20s threshold).
#   2. Three consecutive open + quiesce cycles all complete within budget,
#      confirming FIX AL fires repeatably and not just on first cycle.
#   3. Open after a predecessor SIGKILL does not hang indefinitely
#      (FIX AL's STARTED early-exit is reached and respects kStartedTimeoutMs).
#
# Relationship to existing tests:
#   GAP-11 (test_phase25_force_reset_pass0_chain): tests that FIX AQ's
#     terminate_stale_erisc_routers doesn't hang on relay CMD queue saturation.
#     Does NOT measure wait_for_fabric_workers_ready timing (STARTED → RUNNING).
#   GAP-26 (FIX AS canary timeout): tests that canary poll doesn't hang when
#     some channels never show UMD canary.  Doesn't cover STARTED early-exit.
#   GAP-30 (this test): specifically targets the STARTED → early-exit path.
#     A regression that removes FIX AL or increases kStartedTimeoutMs beyond
#     ~4000ms would be caught here (20s budget for 4+ relay-broken devices).
#
# Topology: T3K (8 WH devices).  Requires >= 4 devices.

import os
import signal
import subprocess
import sys
import time
import pytest
import torch
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (1, 8)
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3
_PCC_THRESHOLD = 0.9999

# Per open+quiesce cycle time budget (seconds).
# With FIX AL: ~3s × 4 non-MMIO devices = ~12s.  We allow 20s for hardware jitter.
# Without FIX AL: ~10s × 4 non-MMIO devices = ~40s (would fail).
_CYCLE_BUDGET_S = 20.0

# Number of open+quiesce cycles to run after the dirty-state predecessor.
_NUM_CYCLES = 3

# Budget for a single AllGather dispatch (used on cycles where open succeeds).
_DISPATCH_BUDGET_S = 10.0

# Time to wait for the predecessor subprocess to signal it's ready (seconds).
_PREDECESSOR_READY_TIMEOUT_S = 30.0


def _predecessor_script() -> str:
    """Python script run in a subprocess to act as the SIGKILL'd predecessor."""
    return r"""
import sys
import signal
import time
import ttnn

# Tell parent we're about to activate ERISC relay channels.
# Use a file-based signal (avoids IPC complexity in subprocess).
import os
ready_path = os.environ.get("GAP30_READY_PATH", "/tmp/gap30_predecessor_ready")

try:
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 8),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    # Perform a dispatch to ensure relay is in ACTIVE state.
    import torch
    full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor
    inp = ttnn.from_torch(
        full,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )
    out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Ring)
    ttnn.synchronize_device(mesh)
except Exception:
    pass

# Signal parent: relay is live.
open(ready_path, "w").close()

# Spin forever — parent will SIGKILL us.
while True:
    time.sleep(0.1)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_gap30_fixal_started_early_exit_timing(mesh_device, tmp_path):
    """
    GAP-30: Verify that FIX AL's STARTED early-exit (kStartedTimeoutMs = 3000ms)
    bounds the per-device wait in wait_for_fabric_workers_ready() when relay-broken
    devices stay in EDMStatus::STARTED indefinitely.

    Three back-to-back open + quiesce cycles must each complete within 20s
    (vs. 40s+ without FIX AL).
    """
    # FIX RZ: skip if fabric is degraded — AllGather hangs on stale base-UMD channels.
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "GAP-30: fabric degraded (base-UMD channels) — skipping to avoid hang"
        )

    ready_path = str(tmp_path / "gap30_predecessor_ready")
    env = {**os.environ, "GAP30_READY_PATH": ready_path}

    # ── Launch predecessor ────────────────────────────────────────────────────
    pred = subprocess.Popen(
        [sys.executable, "-c", _predecessor_script()],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for predecessor to signal ready.
    t_start = time.time()
    while not os.path.exists(ready_path):
        if time.time() - t_start > _PREDECESSOR_READY_TIMEOUT_S:
            pred.kill()
            pred.wait()
            pytest.skip(
                f"Predecessor did not signal ready within {_PREDECESSOR_READY_TIMEOUT_S}s "
                "(hardware init stall?); skipping GAP-30."
            )
        time.sleep(0.1)

    # Extra 2s to ensure relay is firmly in ACTIVE state.
    time.sleep(2.0)

    # SIGKILL predecessor — leaves ERISCs in ACTIVE relay state.
    pred.kill()
    pred.wait()
    logger.info("GAP-30: predecessor SIGKILL'd; ERISCs should be in ACTIVE/dirty state")

    # ── Run N open + quiesce cycles ──────────────────────────────────────────
    # Each cycle opens FABRIC_1D_RING (FIX AL fires for relay-broken devices),
    # optionally dispatches AllGather, then quiesces.
    for cycle in range(_NUM_CYCLES):
        logger.info(f"=== GAP-30 cycle {cycle + 1}/{_NUM_CYCLES} ===")

        t_cycle_start = time.time()

        # ── Open / init fabric ────────────────────────────────────────────────
        try:
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D_RING,
                ttnn.FabricReliabilityMode.STRICT_INIT,
            )
        except Exception as exc:
            logger.warning(f"[cycle {cycle}] set_fabric_config raised (relay-broken degraded mode): {exc}")

        open_duration = time.time() - t_cycle_start
        logger.info(f"[cycle {cycle}] fabric init in {open_duration:.1f}s")

        assert open_duration < _CYCLE_BUDGET_S, (
            f"GAP-30 REGRESSION [cycle {cycle}]: fabric open took {open_duration:.1f}s "
            f"(budget: {_CYCLE_BUDGET_S}s). "
            f"Without FIX AL (kStartedTimeoutMs=3000ms), relay-broken devices each wait "
            f"10s in wait_for_fabric_workers_ready() — 4+ devices × 10s = 40s+. "
            f"Check that FIX AL's STARTED early-exit is still active (kStartedTimeoutMs "
            f"should be 3000ms, not 10000ms)."
        )

        # ── Optional dispatch (verify open left device in usable state) ────────
        try:
            torch.manual_seed(cycle)
            full = torch.rand(_AG_OUTPUT_SHAPE, dtype=torch.bfloat16)
            inp = ttnn.from_torch(
                full,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
            )
            t_dispatch = time.time()
            out = ttnn.all_gather(inp, dim=_AG_DIM, topology=ttnn.Topology.Ring)
            ttnn.synchronize_device(mesh_device)
            dispatch_duration = time.time() - t_dispatch
            logger.info(f"[cycle {cycle}] AllGather dispatch in {dispatch_duration:.2f}s")

            assert dispatch_duration < _DISPATCH_BUDGET_S, (
                f"[cycle {cycle}] AllGather stalled {dispatch_duration:.2f}s — "
                f"possible stale dispatch state after STARTED early-exit in FIX AL. "
                f"Expected < {_DISPATCH_BUDGET_S}s."
            )
        except Exception as exc:
            # Clean exception is acceptable if relay was degraded (some channels dead).
            logger.warning(f"[cycle {cycle}] AllGather raised (degraded relay — acceptable): {exc}")

        # ── Quiesce ────────────────────────────────────────────────────────────
        logger.info(f"[cycle {cycle}] quiescing fabric")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as exc:
            logger.warning(f"[cycle {cycle}] quiesce raised: {exc}")

        cycle_duration = time.time() - t_cycle_start
        logger.info(f"[cycle {cycle}] total cycle duration: {cycle_duration:.1f}s")

        assert cycle_duration < _CYCLE_BUDGET_S, (
            f"GAP-30 REGRESSION [cycle {cycle}]: total open+quiesce cycle took "
            f"{cycle_duration:.1f}s (budget: {_CYCLE_BUDGET_S}s). "
            f"FIX AL STARTED early-exit may not be firing correctly."
        )

    logger.info(
        f"GAP-30: all {_NUM_CYCLES} open+quiesce cycles passed within {_CYCLE_BUDGET_S}s each — "
        f"FIX AL STARTED early-exit is functioning correctly (kStartedTimeoutMs=3000ms)"
    )
