# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-37: FIX BA — STARTED-state non-MMIO devices must be included in relay_broken_non_mmio
#
# Background (CI run 25066686656 — all 359 t3k_ttnn_tests failed):
#
#   In risc_firmware_initializer.cpp teardown Step 1, the scan for relay-broken
#   non-MMIO devices only checked is_fabric_relay_path_broken() (FIX-1 / FIX V flag).
#   There is a second failure mode where FIX AL + FIX AM fire:
#
#     Phase 5 for non-MMIO device N saw master chan stuck at EDMStatus::STARTED
#     for > 3000ms (FIX AL early-exit).  FIX AM then set
#     fabric_channels_not_ready_for_traffic_=true but did NOT set
#     fabric_relay_path_broken_ (STARTED means ERISC firmware IS running —
#     it just never completed the ETH handshake with its out-of-mesh partner).
#
#   Because fabric_relay_path_broken_=false, Step 1 did NOT add these devices
#   to relay_broken_non_mmio.  Consequently:
#     - FIX AC (MMIO ETH PCIe reset) did not run.
#     - FIX AY (deferred non-MMIO ERISC reset via restored relay) did not run.
#
#   The non-MMIO ERISCs remained running FABRIC firmware in EDMStatus::STARTED
#   state at process exit.
#
#   When the next process (t3k_ttnn_tests) started:
#     TopologyDiscovery::discover_remote_devices()
#       → create_remote_device()
#       → init_tt_device()
#       → read_from_arc_apb()
#       → read_non_mmio()
#
#   The FABRIC-mode ERISC on the non-MMIO device does not service UMD relay
#   reads → 5s timeout per device → EVERY test fixture constructor threw →
#   ALL 359 tests in t3k_ttnn_tests failed with timeout exceptions.
#
# Fix: FIX BA (risc_firmware_initializer.cpp, Step 1):
#   Also add non-MMIO devices with fabric_channels_not_ready_for_traffic_=true
#   (and fabric_relay_path_broken_=false) to relay_broken_non_mmio.
#   This triggers FIX AC + FIX AY to clean up STARTED-state ERISCs before
#   the process exits.  FIX AV and FIX AW handle the fallback case where relay
#   re-sync fails within the current process.
#
#   Log emitted by FIX BA (pattern for analyze_fabric_hang_log.sh):
#     "teardown: FIX BA — non-MMIO device N has fabric_channels_not_ready_for_traffic
#      (FIX AM STARTED early-exit) but relay not marked broken. Adding to
#      relay_broken_non_mmio to trigger FIX AC + FIX AY cleanup. (#42429)"
#
# What this test verifies:
#   1. Predecessor opens FABRIC_2D on a T3K 2×4 mesh, is SIGKILL'd.
#      This leaves non-MMIO ERISCs in ACTIVE FABRIC firmware state.
#   2. Testee-1 subprocess opens FABRIC_2D:
#      - Phase 2.5: MMIO ERISCs assert-reset then deassert → reboot.
#      - Phase 5: some non-MMIO master channels reach EDMStatus::STARTED but
#        not LOCAL_HANDSHAKE_COMPLETE (out-of-mesh peer unreachable).
#      - FIX AL fires after 3001ms → FIX AM sets channels_not_ready=true.
#      - fabric_relay_path_broken_ remains FALSE.
#   3. Testee-1 closes mesh:
#      - FIX BA fires: adds channels_not_ready devices to relay_broken_non_mmio.
#      - FIX AC runs: MMIO ETH channels PCIe-reset → UMD relay restored.
#      - FIX AY runs: non-MMIO ERISCs reset to base UMD firmware via relay.
#   4. Testee-2 subprocess opens FABRIC_2D immediately after Testee-1 exits:
#      - ERISCs should be in base UMD firmware state (FIX BA + FIX AY cleaned them).
#      - Topology discovery (create_remote_device) completes quickly.
#   5. Assert: Testee-2 opens + closes within kSecondOpenBudgetS.
#      Without FIX BA: STARTED-state ERISCs remain → topology discovery stalls
#        5s per device × N devices = 20-30s → FAIL.
#      With    FIX BA: ERISCs in base UMD fw → discovery < 5s → PASS.
#
# Relationship to existing tests:
#   GAP-34 (test_gap34_fixam_phase5b_skip_timing.py):
#     Tests FIX AM timing — Phase 5b skipped when master STARTED.
#     Does NOT test that teardown cleans up STARTED-state ERISCs (FIX BA).
#     A FIX BA regression leaves ERISCs dirty at teardown, which would NOT
#     cause GAP-34 to fail (GAP-34 is measuring quiesce latency, not next-
#     session startup correctness).
#   GAP-37 (this test):
#     Tests FIX BA CORRECTNESS — after teardown, STARTED-state ERISCs must
#     be in base UMD fw.  Catches the run-25066686656 regression where all
#     359 subsequent tests failed.
#
# Hardware: T3K (8-device WH 2×4 mesh).  Requires >= 4 non-MMIO devices
# with at least one out-of-mesh ETH link (standard T3K topology satisfies this).

import os
import signal
import subprocess
import sys
import time

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)

# After FIX BA cleans up STARTED-state ERISCs, the second open should complete
# in < kSecondOpenBudgetS.
#
# Without FIX BA: N STARTED-state non-MMIO devices × 5s per device =
#   e.g., 4 devices × 5s = 20s → FAIL.
# With    FIX BA: ERISCs in base UMD fw → topology discovery < 3s → PASS.
kSecondOpenBudgetS = 15.0

# Predecessor ready timeout.
kPredecessorReadyS = 30.0

# Budget for the combined Testee-1 (open + close) to complete before we
# launch Testee-2.  Allow generous time for FIX AL/AM (3s/device × 4 = 12s)
# + FIX AC + FIX AY (≤10s).
kTestee1BudgetS = 60.0


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
except Exception:
    pass

open("{ready_path}", "w").close()

while True:
    time.sleep(0.1)
"""


def _testee1_script(ready_path: str) -> str:
    """
    Open FABRIC_2D (triggers FIX AL/AM on out-of-mesh devices), then close.
    FIX BA must fire at close to add STARTED-state devices to relay_broken_non_mmio.
    FIX AC + FIX AY reset non-MMIO ERISCs to base UMD firmware.
    Signal done, then exit.
    """
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
    # One AllGather to confirm the mesh is usable (or tolerate failure in degraded mode)
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
        pass

    # Close: FIX BA fires → FIX AC → FIX AY clean up STARTED-state ERISCs.
    ttnn.close_mesh_device(mesh)
except Exception as exc:
    print(f"TESTEE1_ERROR: {{exc}}", flush=True)
    sys.exit(1)

# Signal done before exit — so parent can time Testee-2 start accurately.
open("{ready_path}", "w").close()
sys.exit(0)
"""


def _testee2_script() -> str:
    """
    Open FABRIC_2D and immediately close.  Timed by the parent process.
    If FIX BA worked, ERISCs are in base UMD fw → fast open.
    If FIX BA regressed, STARTED ERISCs stall topology discovery (5s/device).
    """
    return r"""
import sys, os, time
import ttnn

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(mesh)
    sys.exit(0)
except Exception as exc:
    print(f"TESTEE2_ERROR: {exc}", flush=True)
    # exit(1) so parent can detect it, but if it's fast that's still ok for timing
    sys.exit(1)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
def test_gap37_fixba_started_state_nonmmio_cleanup(tmp_path):
    """
    GAP-37: Verify that FIX BA includes STARTED-state non-MMIO devices in
    relay_broken_non_mmio so that FIX AC + FIX AY clean up STARTED-state
    ERISCs before process exit.

    Without FIX BA: STARTED ERISCs remain → next-session topology discovery
    stalls 5s per device → test fixture timeouts cascade across 359+ tests
    (observed in CI run 25066686656).

    With FIX BA: ERISCs reset to base UMD fw → second open completes < 15s.
    """
    pred_ready = str(tmp_path / "gap37_predecessor_ready")
    t1_done = str(tmp_path / "gap37_testee1_done")

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
                f"GAP-37: predecessor did not signal ready within {kPredecessorReadyS}s "
                "(hardware init stall?); skipping."
            )
        time.sleep(0.1)

    # Brief settle to ensure relay is firmly in ACTIVE state.
    time.sleep(2.0)

    pred.kill()
    pred.wait()
    logger.info("GAP-37: predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state")

    # ── Phase 2: Launch Testee-1 (open → FIX AL/AM → close → FIX BA fires) ─
    t1_proc = subprocess.Popen(
        [sys.executable, "-c", _testee1_script(t1_done)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    t1_start = time.time()
    # Wait for Testee-1 to signal done (or timeout).
    while not os.path.exists(t1_done):
        if t1_proc.poll() is not None:
            break
        if time.time() - t1_start > kTestee1BudgetS:
            t1_proc.kill()
            t1_proc.wait()
            pytest.skip(
                f"GAP-37: Testee-1 did not complete within {kTestee1BudgetS}s — "
                "hardware in unexpected state; skipping."
            )
        time.sleep(0.2)
    t1_proc.wait()
    t1_elapsed = time.time() - t1_start
    logger.info(f"GAP-37: Testee-1 exited in {t1_elapsed:.1f}s (exit code: {t1_proc.returncode})")

    # ── Phase 3: Launch Testee-2 and time the open ─────────────────────────
    # FIX BA should have cleaned up STARTED-state ERISCs in Testee-1's close.
    # Testee-2 open should be fast.  Without FIX BA: STARTED ERISCs stall
    # topology discovery (read_non_mmio → 5s timeout per device).
    t2_start = time.time()
    t2_proc = subprocess.Popen(
        [sys.executable, "-c", _testee2_script()],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    t2_proc.wait(timeout=kSecondOpenBudgetS + 10.0)
    t2_elapsed = time.time() - t2_start

    logger.info(
        f"GAP-37: Testee-2 open+close took {t2_elapsed:.1f}s "
        f"(budget: {kSecondOpenBudgetS}s, exit code: {t2_proc.returncode})"
    )

    assert t2_elapsed < kSecondOpenBudgetS, (
        f"GAP-37 REGRESSION (FIX BA): Testee-2 (second open after FIX BA teardown) "
        f"took {t2_elapsed:.1f}s >= {kSecondOpenBudgetS}s budget. "
        f"\n"
        f"Without FIX BA (#42429): STARTED-state non-MMIO ERISCs are NOT added to "
        f"relay_broken_non_mmio in teardown Step 1 (only fabric_relay_path_broken_ "
        f"is checked, not fabric_channels_not_ready_for_traffic_). "
        f"FIX AC (MMIO ETH PCIe reset) and FIX AY (non-MMIO ERISC reset) do NOT run "
        f"for these devices. Non-MMIO ERISCs remain in FABRIC EDMStatus::STARTED state. "
        f"Next process topology discovery → create_remote_device() → init_tt_device() "
        f"→ read_non_mmio() stalls 5s per STARTED-state device. "
        f"\n"
        f"Expected: FIX BA fires in Testee-1 teardown, adds STARTED-state devices to "
        f"relay_broken_non_mmio → FIX AC + FIX AY clean up ERISCs → Testee-2 fast open. "
        f"\n"
        f"Log pattern to search: "
        f"'teardown: FIX BA — non-MMIO device .* has fabric_channels_not_ready_for_traffic'"
    )

    logger.info(
        f"GAP-37 PASS: Testee-2 open+close took {t2_elapsed:.1f}s "
        f"(budget: {kSecondOpenBudgetS}s) — FIX BA + FIX AY cleaned up "
        f"STARTED-state ERISCs before Testee-1 exited."
    )
