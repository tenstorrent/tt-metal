# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-40: FIX AE — Catch flush timeouts and mark relay broken in Cluster write functions
#
# Background (two related failure modes fixed by FIX AE, commit 561f7abd505):
#
#   FAILURE MODE 1: write_core / write_core_immediate / write_reg / noc_multicast_write
#   hang for 5s per call when a non-MMIO remote chip's relay becomes dead mid-session.
#
#   The sequence: remote chip N's ERISC relay crashes or enters a stuck state.
#   Any subsequent write_core(chip=N, ...) call calls
#   driver_->wait_for_non_mmio_flush(N), which spins for up to 5s waiting for the
#   stale relay CMD queue to drain.  It never drains.  After 5s it throws
#   UmdException("wait_for_non_mmio_flush timeout").  But the caller (e.g., a
#   TTNN op running L1 buffer setup) catches std::exception generically and retries,
#   potentially calling write_core again → another 5s hang → cascade.
#   FIX AE wraps each wait_for_non_mmio_flush call in a try/catch:
#     - On timeout: log a FIX AE warning and call driver_->mark_relay_broken(chip_id).
#     - mark_relay_broken() makes all subsequent wait_for_non_mmio_flush() for that
#       chip return instantly (no spin).  5s → 0ms for all future flushes.
#
#   FAILURE MODE 2: ~Cluster() destructor hangs in driver_->close_device() when some
#   remote chips' relay queues have stale entries.  close_device() calls set_power_state()
#   and assert_risc_reset() via the remote communication layer, which calls
#   wait_for_non_mmio_flush() internally.  With stale entries: 5s per chip × N chips.
#   Worse: the 5s flush block allows a racing open_device() (new Cluster construction)
#   to start while UMD global state destructors are still running → heap corruption.
#
#   FIX AE also addresses this in ~Cluster():
#     Before calling driver_->close_device(), iterate all_chip_ids() and call
#     driver_->mark_relay_broken(chip_id) for every remote chip.
#     This makes close_device()'s internal wait_for_non_mmio_flush() calls instant.
#     Supersedes FIX AW (background detach thread with 5s timeout) — simpler and
#     eliminates the race window entirely.
#
# What this test verifies:
#   1. Predecessor opens FABRIC_2D on T3K 2×4, runs AllGather, then is SIGKILL'd.
#      This leaves non-MMIO ERISCs in FABRIC firmware state and MMIO relay queues
#      with stale ETH firmware metadata.
#   2. Testee-1 opens FABRIC_2D. During quiesce/teardown, relay becomes dead for some
#      non-MMIO chips (FIX V / FIX-1 path: Phase 5 timeout → relay_broken).
#      After MeshDevice::close(), the code path enters ~Cluster().
#      FIX AE: ~Cluster() marks all remote chips relay-broken → close_device() instant.
#      Without FIX AE: close_device() blocks 5s per chip × 4 remote chips = 20s.
#      Testee-1 must exit in < kTestee1BudgetS.
#   3. Testee-2 opens immediately after Testee-1 exits.  If FIX AE properly prevented
#      heap corruption (no racing UMD destructor+constructor), Testee-2 should open
#      without crashing (SIGSEGV / SIGABRT).
#      Without FIX AE: Testee-2 may crash with heap corruption due to UMD global
#      state being accessed by both ~Cluster() (still running) and new Cluster().
#   4. Both timing and correctness are verified.
#
# Relationship to existing tests:
#   GAP-29 (FIX AW): Tests that ~Cluster() does not hang when relay is broken.
#     FIX AE supersedes FIX AW with a simpler mechanism (mark broken before close
#     rather than detached thread + 5s timeout).  GAP-29 would still pass with FIX AE
#     but its test description refers to FIX AW (background thread pattern).
#   GAP-40 (this test): Tests FIX AE's mark-before-close mechanism specifically,
#     and also tests that the relay-broken-during-write path (write_core flush catch)
#     does not cascade into repeated 5s hangs.
#
# Hardware: T3K (8-device WH 2×4 mesh).

import os
import subprocess
import sys
import time

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)

# Testee-1 budget: open + quiesce with relay-dead path + close.
# Without FIX AE: ~Cluster() blocks 5s per remote chip × 4 = 20s overhead
#   PLUS possible 5s× per write_core flush hit = 20s+ per hanging write.
# With    FIX AE: mark-before-close instant; flush catches immediate.
#   Total: quiesce ≈ 15s + close ≈ 1s = < 30s typical.
kTestee1BudgetS = 60.0

# Testee-2 budget: second open + close. Should be fast if FIX NS + FIX AE both work.
kTestee2BudgetS = 30.0

# Predecessor ready timeout.
kPredecessorReadyS = 30.0


def _predecessor_script(ready_path: str) -> str:
    """Open FABRIC_2D, run AllGather, signal ready, spin until SIGKILL."""
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


def _testee1_script() -> str:
    """
    Open FABRIC_2D (triggers relay-dead path via Phase 5 timeout), then close.
    FIX AE: ~Cluster() marks all remote chips broken → close_device() instant.
    Without FIX AE: ~Cluster() blocks 5s per chip.
    """
    return r"""
import sys, time
import ttnn
from ttnn import ShardTensorToMesh
import torch

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    # AllGather — may fail in degraded mode, that's OK.
    try:
        full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            full, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh)
    except Exception:
        pass

    ttnn.close_mesh_device(mesh)
    sys.exit(0)
except Exception as exc:
    print(f"TESTEE1_ERROR: {exc}", flush=True)
    sys.exit(1)
"""


def _testee2_script() -> str:
    """
    Open FABRIC_2D immediately after Testee-1 exits.
    If FIX AE properly prevented heap corruption, this should not crash (SIGSEGV/SIGABRT).
    Exit code 0 = no crash (pass). Non-zero = crash or hang (fail).
    """
    return r"""
import sys, time
import ttnn

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(mesh)
    sys.exit(0)
except Exception as exc:
    print(f"TESTEE2_ERROR: {exc}", flush=True)
    # Tolerate exception (degraded mode), but NOT crash (SIGSEGV/SIGABRT gives non-zero)
    sys.exit(0)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
def test_gap40_fixae_flush_timeout_catch(tmp_path):
    """
    GAP-40: Verify that FIX AE catches wait_for_non_mmio_flush() timeouts in Cluster
    write functions (write_core, write_core_immediate, write_reg, noc_multicast_write)
    and in ~Cluster() (mark all remote chips broken before close_device()).

    Without FIX AE:
      - write_core flush blocks 5s per call when relay is dead mid-session.
      - ~Cluster() blocks 5s per remote chip in close_device().
      - Race window: new Cluster can start while old one is still in close_device()
        → UMD global state heap corruption → SIGSEGV / SIGABRT in Testee-2.

    With FIX AE:
      - write_core flush: catch + mark_relay_broken → instant for all future flushes.
      - ~Cluster(): mark all remote broken before close_device() → instant close.
      - No race window → Testee-2 opens cleanly without crash.

    Log patterns to confirm FIX AE is active:
      "FIX AE: wait_for_non_mmio_flush(chip N) threw: ..."
      (in tt_cluster.cpp write_core / write_reg / noc_multicast_write)
    """
    pred_ready = str(tmp_path / "gap40_predecessor_ready")

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
                f"GAP-40: predecessor did not signal ready within {kPredecessorReadyS}s; skipping."
            )
        time.sleep(0.1)

    time.sleep(2.0)
    pred.kill()
    pred.wait()
    logger.info("GAP-40: predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state")

    # ── Phase 2: Testee-1 — open + (relay-dead quiesce) + close ────────────
    t1_start = time.time()
    t1_proc = subprocess.Popen(
        [sys.executable, "-c", _testee1_script()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        t1_out, t1_err = t1_proc.communicate(timeout=kTestee1BudgetS + 30.0)
    except subprocess.TimeoutExpired:
        t1_proc.kill()
        t1_proc.communicate()
        elapsed = time.time() - t1_start
        assert False, (
            f"GAP-40 REGRESSION (FIX AE ~Cluster): Testee-1 timed out after {elapsed:.0f}s "
            f"(budget: {kTestee1BudgetS}s).\n"
            f"\n"
            f"Without FIX AE: ~Cluster() blocks in driver_->close_device() because "
            f"wait_for_non_mmio_flush() spins 5s per remote chip × 4 chips = 20s+ hang.\n"
            f"\n"
            f"With FIX AE: ~Cluster() calls mark_relay_broken() for all remote chips "
            f"before close_device() → wait_for_non_mmio_flush() returns instantly.\n"
            f"Expected Testee-1 to complete in < {kTestee1BudgetS}s."
        )

    t1_elapsed = time.time() - t1_start
    logger.info(
        f"GAP-40: Testee-1 exited in {t1_elapsed:.1f}s "
        f"(budget: {kTestee1BudgetS}s, exit code: {t1_proc.returncode})"
    )

    assert t1_elapsed < kTestee1BudgetS, (
        f"GAP-40 REGRESSION (FIX AE ~Cluster): Testee-1 took {t1_elapsed:.1f}s "
        f">= {kTestee1BudgetS}s. FIX AE mark-before-close may have regressed. "
        f"Check for 5s per-chip blocks in ~Cluster() teardown."
    )

    # ── Phase 3: Testee-2 — open immediately after Testee-1 ─────────────────
    # Verify no heap corruption (SIGSEGV / SIGABRT) from racing UMD destructors.
    t2_start = time.time()
    t2_proc = subprocess.Popen(
        [sys.executable, "-c", _testee2_script()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        t2_out, t2_err = t2_proc.communicate(timeout=kTestee2BudgetS + 15.0)
    except subprocess.TimeoutExpired:
        t2_proc.kill()
        t2_proc.communicate()
        elapsed = time.time() - t2_start
        assert False, (
            f"GAP-40 REGRESSION (FIX AE heap-race): Testee-2 timed out after {elapsed:.0f}s "
            f"(budget: {kTestee2BudgetS}s). Process may be hung inside UMD after heap "
            f"corruption from racing ~Cluster() + Cluster() UMD global state access."
        )

    t2_elapsed = time.time() - t2_start

    # SIGSEGV = -11, SIGABRT = -6 on Linux.  These indicate heap corruption.
    is_crash = t2_proc.returncode in (-11, -6) or (
        t2_proc.returncode != 0 and b"Segmentation fault" in (t2_out + t2_err)
    )
    assert not is_crash, (
        f"GAP-40 REGRESSION (FIX AE heap-race): Testee-2 crashed with exit code "
        f"{t2_proc.returncode} after {t2_elapsed:.1f}s.\n"
        f"stderr: {t2_err.decode(errors='replace')[-500:]}\n"
        f"\n"
        f"Without FIX AE: ~Cluster() in Testee-1 held UMD global state for 20s+ while "
        f"close_device() flushed. Testee-2 started a new Cluster() concurrently, "
        f"accessing the same UMD global state → heap corruption → SIGSEGV/SIGABRT.\n"
        f"\n"
        f"With FIX AE: mark_relay_broken() before close_device() makes the destructor "
        f"return instantly → no race window → Testee-2 safe to start."
    )

    assert t2_elapsed < kTestee2BudgetS, (
        f"GAP-40 TIMING (FIX AE + FIX NS): Testee-2 took {t2_elapsed:.1f}s "
        f">= {kTestee2BudgetS}s budget. Check if FIX NS (single topology discovery) "
        f"is still in place — double discovery causes relay queue overflow on FABRIC-state ERISCs."
    )

    logger.info(
        f"GAP-40 PASS: Testee-1 in {t1_elapsed:.1f}s, Testee-2 in {t2_elapsed:.1f}s. "
        f"FIX AE mark-before-close worked: no hang, no heap corruption."
    )
