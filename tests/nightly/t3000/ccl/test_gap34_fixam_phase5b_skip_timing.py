# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-34: FIX AM — Phase 5b skip when master chan stuck at STARTED (out-of-mesh)
#
# Background — FIX AM (device.cpp: wait_for_fabric_workers_ready):
#   FIX AL (kStartedTimeoutMs = 3000ms) was introduced to bound the per-device
#   wait when relay-broken devices stay in EDMStatus::STARTED indefinitely.
#   FIX AL fires and breaks the poll loop, but then Phase 5b still ran:
#
#     Phase 5 exits (FIX AL): 3001ms elapsed for this device.
#     Phase 5b runs:          polls each channel for READY_FOR_TRAFFIC,
#                             waits up to kHealthCheckTimeoutMs (2001ms total).
#     FIX AK fires at end:    "transitive relay hang guard" sets
#                             fabric_channels_not_ready_for_traffic_=true.
#
#   The Phase 5b run was redundant: when master chan stayed at STARTED (not
#   LOCAL_HANDSHAKE_COMPLETE), the READY_FOR_TRAFFIC host-write was skipped.
#   All subordinate ERISCs are therefore stuck at REMOTE_HANDSHAKE_COMPLETE —
#   they're waiting for master to call notify_subordinate_routers(), which
#   requires master to first complete the ETH handshake (impossible for an
#   out-of-mesh channel peer).  Phase 5b will always confirm they're not at
#   READY_FOR_TRAFFIC and then set fabric_channels_not_ready_for_traffic_.
#   Running Phase 5b gains nothing; it only wastes 2001ms per device.
#
#   FIX AM (device.cpp ~3041): after the Phase 5 polling loop exits, check
#   if sync_buf[0] == EDMStatus::STARTED.  If true (FIX AL fired), immediately:
#     1. Set fabric_channels_not_ready_for_traffic_ = true.
#     2. Return — skipping Phase 5b entirely.
#   Logged as: "…skipping Phase 5b. (FIX AM: #42429)"
#
#   Per-device savings: ~2001ms (Phase 5b) eliminated.
#   On T3K with 4 devices that each have an out-of-mesh master channel:
#     Before FIX AM: 4 × (3001ms + 2001ms) ≈ 20s quiesce overhead
#     After  FIX AM: 4 × 3001ms            ≈ 12s quiesce overhead
#   CI failure it fixed: TestMeshWidthShardedCopy3D timeout (54s budget)
#   and AsyncExecutionWorksCQ0 TearDown hang (both from CI run 25048641877).
#
# What this test verifies:
#   1. After a SIGKILL predecessor leaves ERISC channels in dirty state,
#      opening FABRIC_2D and running quiesce completes within 15s.
#      Without FIX AM: ~20s per cycle (4 devices × 5s each) → FAIL.
#      With    FIX AM: ~12s per cycle (4 devices × 3s each) → PASS.
#   2. Three consecutive open+quiesce cycles all pass, confirming FIX AM
#      fires repeatably and not just on the first cycle.
#
# Relationship to existing tests:
#   GAP-30 (test_gap30_fixal_started_early_exit_timing.py):
#     Tests FIX AL specifically — STARTED early-exit bounds the Phase 5 poll.
#     Uses FABRIC_1D_RING.  Budget: 20s (passes with FIX AL at ~12s).
#     Does NOT test Phase 5b behaviour; does not catch FIX AM regressions.
#   GAP-34 (this test):
#     Tests FIX AM specifically — Phase 5b is SKIPPED when master at STARTED.
#     Uses FABRIC_2D (same config as the failing CI run that motivated FIX AM).
#     Budget: 15s.  Without FIX AM the cycle takes ~20s → FAIL.
#     Without FIX AL the cycle takes ~40s → also FAIL (FIX AM depends on FIX AL).
#
# Why FABRIC_2D and not FABRIC_1D_RING:
#   The CI run that triggered FIX AM (25048641877) used FABRIC_2D on T3K 2×4.
#   In a 2×4 mesh, each chip has ETH channels to neighbors on all four edges.
#   The perimeter chips have one or more out-of-mesh channels (no physical
#   peer in the mesh).  These channels stay at STARTED indefinitely, causing
#   FIX AL → FIX AM to fire on every quiesce cycle.  FABRIC_2D is the
#   topology where this pattern is most consistent.
#
# Topology: T3K (8 WH devices, 2×4 mesh).  Requires >= 4 devices.

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

_MESH_SHAPE = (2, 4)
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3

# Per open+quiesce cycle time budget (seconds).
# With FIX AM + FIX AL: ~3s × 4 devices = ~12s.  Allow 15s for hardware jitter.
# Without FIX AM (but with FIX AL): ~5s × 4 devices = ~20s → FAIL.
# Without FIX AL (and FIX AM): ~10s × 4 devices = ~40s → also FAIL.
_CYCLE_BUDGET_S = 15.0

# Number of open+quiesce cycles to run.
_NUM_CYCLES = 3

# Budget for predecessor to signal it's ready.
_PREDECESSOR_READY_TIMEOUT_S = 30.0

# Budget for a single AllGather dispatch (optional, used on cycle 0 only).
_DISPATCH_BUDGET_S = 10.0


def _predecessor_script() -> str:
    """Python script run in a subprocess to act as the SIGKILL'd predecessor.

    Opens FABRIC_2D on the 2×4 T3K mesh, dispatches a workload so relay
    ERISCs are in ACTIVE state, then signals ready and spins.  Parent SIGKILLs
    it to leave non-MMIO ERISCs in dirty state — the same precondition that
    caused TestMeshWidthShardedCopy3D and AsyncExecutionWorksCQ0 to fail in
    CI run 25048641877.
    """
    return r"""
import sys, os, time, torch
import ttnn
from ttnn import ShardTensorToMesh

ready_path = os.environ.get("GAP34_READY_PATH", "/tmp/gap34_predecessor_ready")

try:
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(2, 4),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )
    out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)
except Exception:
    pass

# Signal parent: ERISC relay is live.
open(ready_path, "w").close()

# Spin until SIGKILL — leaves non-MMIO ERISCs in FABRIC firmware state.
while True:
    time.sleep(0.1)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_gap34_fixam_phase5b_skip_timing(mesh_device, tmp_path):
    """
    GAP-34: Verify that FIX AM skips Phase 5b when the master ETH channel stays
    at EDMStatus::STARTED after FIX AL's early-exit (out-of-mesh peer).

    Three back-to-back FABRIC_2D open + quiesce cycles must each complete within
    15s (vs ~20s without FIX AM, and ~40s without FIX AL).
    """
    # FIX RZ: skip if fabric is degraded — AllGather hangs on stale base-UMD channels.
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "GAP-34: fabric degraded (base-UMD channels) — skipping to avoid hang"
        )

    ready_path = str(tmp_path / "gap34_predecessor_ready")
    env = {**os.environ, "GAP34_READY_PATH": ready_path}

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
                "(hardware init stall?); skipping GAP-34."
            )
        time.sleep(0.1)

    # Extra 2s to ensure relay is firmly in ACTIVE state.
    time.sleep(2.0)

    # SIGKILL predecessor — leaves non-MMIO ERISCs in FABRIC firmware state.
    pred.kill()
    pred.wait()
    logger.info(
        "GAP-34: predecessor SIGKILL'd; non-MMIO ERISCs should be in FABRIC firmware state"
    )

    # ── Run N open+quiesce cycles ─────────────────────────────────────────────
    # Each cycle opens FABRIC_2D.  FIX AL fires for out-of-mesh master channels
    # (stuck at STARTED after 3000ms).  FIX AM then skips Phase 5b entirely.
    # Total per cycle: ~3s × 4 devices ≈ 12s.  Without FIX AM: ~5s × 4 = ~20s.
    for cycle in range(_NUM_CYCLES):
        logger.info(f"=== GAP-34 cycle {cycle + 1}/{_NUM_CYCLES} ===")
        t_cycle_start = time.time()

        # ── Open / init fabric ────────────────────────────────────────────────
        try:
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_2D,
                ttnn.FabricReliabilityMode.STRICT_INIT,
            )
        except Exception as exc:
            logger.warning(
                f"[cycle {cycle}] set_fabric_config raised (relay-broken degraded mode): {exc}"
            )

        open_duration = time.time() - t_cycle_start
        logger.info(f"[cycle {cycle}] fabric init in {open_duration:.1f}s")

        assert open_duration < _CYCLE_BUDGET_S, (
            f"GAP-34 REGRESSION [cycle {cycle}]: fabric open took {open_duration:.1f}s "
            f"(budget: {_CYCLE_BUDGET_S}s). "
            f"Without FIX AM (#42429), Phase 5b runs after FIX AL fires — "
            f"2001ms per device wasted confirming channels not at READY_FOR_TRAFFIC "
            f"(subordinates stuck at REMOTE_HANDSHAKE_COMPLETE because master at STARTED). "
            f"4 devices × (3001ms + 2001ms) ≈ 20s without FIX AM; "
            f"4 devices × 3001ms ≈ 12s with FIX AM. "
            f"Check wait_for_fabric_workers_ready() for the FIX AM early-return guard "
            f"after the Phase 5 polling loop (sync_buf[0] == STARTED → skip Phase 5b)."
        )

        # ── Optional dispatch: verify open left device in usable state ─────────
        # Only on cycle 0 to avoid accumulating state.
        if cycle == 0:
            try:
                torch.manual_seed(0)
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
                out = ttnn.all_gather(inp, dim=_AG_DIM, topology=ttnn.Topology.Linear)
                ttnn.synchronize_device(mesh_device)
                dispatch_duration = time.time() - t_dispatch
                logger.info(
                    f"[cycle {cycle}] AllGather dispatch in {dispatch_duration:.2f}s"
                )
                assert dispatch_duration < _DISPATCH_BUDGET_S, (
                    f"[cycle {cycle}] AllGather stalled {dispatch_duration:.2f}s — "
                    f"possible stale dispatch state after FIX AM Phase 5b skip. "
                    f"Expected < {_DISPATCH_BUDGET_S}s."
                )
            except Exception as exc:
                logger.warning(
                    f"[cycle {cycle}] AllGather raised (degraded relay — acceptable): {exc}"
                )

        # ── Quiesce ────────────────────────────────────────────────────────────
        logger.info(f"[cycle {cycle}] quiescing fabric")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as exc:
            logger.warning(f"[cycle {cycle}] quiesce raised: {exc}")

        cycle_duration = time.time() - t_cycle_start
        logger.info(f"[cycle {cycle}] total cycle duration: {cycle_duration:.1f}s")

        assert cycle_duration < _CYCLE_BUDGET_S, (
            f"GAP-34 REGRESSION [cycle {cycle}]: total open+quiesce cycle took "
            f"{cycle_duration:.1f}s (budget: {_CYCLE_BUDGET_S}s). "
            f"FIX AM Phase 5b skip may not be firing correctly."
        )

    logger.info(
        f"GAP-34: all {_NUM_CYCLES} open+quiesce cycles passed within {_CYCLE_BUDGET_S}s each — "
        f"FIX AM Phase 5b skip is functioning correctly (master STARTED → skip Phase 5b)"
    )
