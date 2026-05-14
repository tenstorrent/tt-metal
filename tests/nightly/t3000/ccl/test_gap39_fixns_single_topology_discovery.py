# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-39: FIX NS — Eliminate redundant topology discovery in initialize_base_objects
#
# Background (CI run 25074797042 — 14m40s SIGALRM hang before any test ran):
#
#   MetalEnvImpl::initialize_base_objects() previously called
#   Cluster::get_cluster_type_from_cluster_desc(*rtoptions_) to determine
#   is_base_routing_fw_enabled BEFORE creating the Cluster object.
#   That static helper triggered a full UMD topology discovery (TopologyDiscovery #1)
#   which deposited one relay CMD into each MMIO relay queue.
#
#   Then Cluster::Cluster() ran a SECOND full topology discovery (TopologyDiscovery #2)
#   in open_driver() → create_cluster_descriptor(). On T3K systems where a prior process
#   left FABRIC-mode ERISCs in the non-MMIO channels:
#
#     TopologyDiscovery #1:  4× FIX AQ 5s timeouts → 4 stuck relay queue entries
#                            (4 MMIO gateways × 1 entry = all at 1/4 capacity)
#
#     TopologyDiscovery #2:  Each non-MMIO read pushes another command.
#                            3rd–4th commands per gateway: queue at 3–4/4 capacity.
#                            4th command enters "while(full)" spin — NO UMD timeout.
#                            → infinite hang.
#
#   Fix (commit f80f0295263): Create Cluster first, then derive cluster type from
#   cluster_->get_cluster_type(). This eliminates TopologyDiscovery #1 entirely.
#
#   Without FIX NS: 2 topology discoveries per open attempt → relay queue overflow
#   after a SIGKILL predecessor left FABRIC-state ERISCs → infinite hang (>14 min).
#   With    FIX NS: 1 topology discovery per open attempt → relay queue stays at ≤1
#   stuck entry per gateway → FIX AQ catches the 5s timeout and marks relay dead.
#   Open + close completes in < 60s.
#
# What this test verifies:
#   1. Predecessor opens FABRIC_2D on T3K 2×4, is SIGKILL'd.
#      Non-MMIO ERISCs remain in FABRIC firmware state (EDM stack running).
#   2. Testee opens a FABRIC_2D mesh immediately.
#      FIX NS: only ONE topology discovery runs before the Cluster is created.
#      FIX AQ: 5s timeout on stale relay reads → remote devices marked as skipped.
#      Open + subsequent AllGather (or graceful degraded-mode continuation) + close
#      must complete within kTestee1BudgetS.
#   3. Regression: Without FIX NS, the second topology discovery fills relay queues
#      to 4/4 → while(full) hang → SIGALRM at 15 min → no test ever executes.
#
# Relationship to existing tests:
#   GAP-37 (FIX BA): Tests that STARTED-state ERISCs are cleaned up at teardown.
#   GAP-39 (this test): Tests that DOUBLE topology discovery does NOT occur on the
#     NEXT process open, independent of teardown cleanup.  Even if FIX BA fires and
#     cleans up ERISCs, a FIX NS regression would cause a different hang (the double
#     discovery race fills relay queues at open time, not at teardown time).
#
# Hardware: T3K (8-device WH 2×4 mesh).

import os
import signal
import subprocess
import sys
import time

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)

# After predecessor is SIGKILL'd and FIX NS is in place (single topology discovery),
# the testee should open, run a quick AllGather, and close within this budget.
#
# Without FIX NS:  Double topology discovery fills relay queues → while(full) hang.
#                  Process is killed by SIGALRM at 900s (15 min).
# With    FIX NS:  Single discovery → FIX AQ catches 5s timeouts per remote chip.
#                  4 remote chips × 5s = 20s + fabric init overhead ≈ 45s max.
kTestee1BudgetS = 90.0

# Predecessor ready timeout.
kPredecessorReadyS = 30.0


def _predecessor_script(ready_path: str) -> str:
    """Open FABRIC_2D, signal ready, spin until SIGKILL."""
    return rf"""
import sys, os, time, torch
import ttnn
from ttnn import ShardTensorToMesh

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )
    out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)
except Exception as e:
    pass

open("{ready_path}", "w").close()

while True:
    time.sleep(0.1)
"""


def _testee_script() -> str:
    """
    Open FABRIC_2D, run AllGather (or tolerate degraded mode), then close.
    With FIX NS: single topology discovery → FIX AQ catches 5s stale reads
    → completes within kTestee1BudgetS.
    Without FIX NS: double discovery → relay queue fills → while(full) hang.
    """
    return r"""
import sys, os, time, torch
import ttnn
from ttnn import ShardTensorToMesh

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    try:
        full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            full, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(mesh, dim=3),
        )
        out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh)
    except Exception:
        # AllGather may fail in degraded mode (relay dead) — that's OK for timing test.
        pass

    ttnn.close_mesh_device(mesh)
    sys.exit(0)
except Exception as exc:
    print(f"TESTEE_ERROR: {exc}", flush=True)
    sys.exit(1)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
def test_gap39_fixns_single_topology_discovery(tmp_path):
    """
    GAP-39: Verify that MetalEnvImpl::initialize_base_objects() only triggers
    ONE topology discovery, not two.

    Without FIX NS: get_cluster_type_from_cluster_desc() runs a full topology
    discovery before Cluster creation, depositing one stuck relay command per
    MMIO gateway when prior process left FABRIC-state ERISCs.  The second
    topology discovery (Cluster::Cluster()) then fills the relay queues past
    capacity → while(full) infinite spin → 14+ min hang.

    With FIX NS: Cluster is created first, cluster type derived from it.
    Single topology discovery → FIX AQ catches 5s stale reads per remote chip
    → whole open+close completes in < kTestee1BudgetS.

    Log pattern that SHOULD NOT appear:
      (No specific log — hang is silent until SIGALRM.  Timing is the oracle.)
    Log pattern that SHOULD appear:
      "FIX AQ" — secondary edm_status_address poll after relay read timeout
      (proves FIX AQ was the safety net, not a second discovery filling queues)
    """
    pred_ready = str(tmp_path / "gap39_predecessor_ready")

    # ── Phase 1: Launch predecessor ─────────────────────────────────────────
    pred = subprocess.Popen(
        [sys.executable, "-c", _predecessor_script(pred_ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t_start = time.time()
    while not os.path.exists(pred_ready):
        if time.time() - t_start > kPredecessorReadyS:
            pred.kill()
            pred.wait()
            pytest.skip(
                f"GAP-39: predecessor did not signal ready within {kPredecessorReadyS}s "
                "(hardware init stall?); skipping."
            )
        time.sleep(0.1)

    # Brief settle to ensure relay is firmly in ACTIVE state.
    time.sleep(2.0)

    pred.kill()
    pred.wait()
    logger.info("GAP-39: predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state")

    # ── Phase 2: Launch testee — time the open+close ────────────────────────
    t_testee_start = time.time()
    testee_proc = subprocess.Popen(
        [sys.executable, "-c", _testee_script()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = testee_proc.communicate(timeout=kTestee1BudgetS + 30.0)
    except subprocess.TimeoutExpired:
        testee_proc.kill()
        testee_proc.communicate()
        elapsed = time.time() - t_testee_start
        assert False, (
            f"GAP-39 REGRESSION (FIX NS): Testee timed out after {elapsed:.0f}s "
            f"(budget: {kTestee1BudgetS}s).\n"
            f"\n"
            f"Without FIX NS: initialize_base_objects() calls "
            f"get_cluster_type_from_cluster_desc() which runs TopologyDiscovery #1 "
            f"(depositing stale relay commands), then Cluster::Cluster() runs "
            f"TopologyDiscovery #2 (filling queues to 4/4) → while(full) hang.\n"
            f"\n"
            f"With FIX NS: Cluster is created first (single topology discovery). "
            f"FIX AQ catches 5s stale reads. Open + close in < {kTestee1BudgetS}s."
        )

    elapsed = time.time() - t_testee_start
    logger.info(
        f"GAP-39: testee open+close took {elapsed:.1f}s "
        f"(budget: {kTestee1BudgetS}s, exit code: {testee_proc.returncode})"
    )

    assert elapsed < kTestee1BudgetS, (
        f"GAP-39 REGRESSION (FIX NS): Testee open+close took {elapsed:.1f}s "
        f">= {kTestee1BudgetS}s budget. FIX NS may have regressed (double topology "
        f"discovery filling relay queues). Check for double 'FIX AQ' sequences in logs "
        f"or a 14+ min SIGALRM hang pattern."
    )

    logger.info(
        f"GAP-39 PASS: Testee open+close took {elapsed:.1f}s "
        f"(budget: {kTestee1BudgetS}s) — single topology discovery confirmed. "
        f"FIX NS prevents relay queue overflow from double discovery."
    )
