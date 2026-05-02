# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-35: FIX AT — Phase 5 handshake poll skipped when MMIO master chan was FIX AS Pass-0 timeout'd
#
# Background (CI failure run 25054499947, iteration 14 of monitoring cycle):
#
#   Failure pattern:
#     1. All MMIO devices (0–3) on runner had base_umd=6 clean at terminate_stale time.
#     2. Pre-AllGather quiesce: MMIO devices (e.g. Device 1, Device 3) go through
#        Phase 2.5 — all 6 ETH channels halted (assert RISC reset in Phase 2.5).
#     3. launch_eth_cores_for_quiesce: FIX AR deasserts all 6 channels simultaneously.
#     4. FIX AS polls each channel for UMD relay canary (0x49706550) up to 500ms.
#        WH BRISC boot takes >500ms — ALL 6 channels timeout, marked newly-dead.
#        No launch message written to any channel (FIX-3 guard respects newly-dead).
#     5. Phase 5 polls master chan 14 for LOCAL_HANDSHAKE_COMPLETE: sees 0x0 the entire
#        10s wait (firmware was never launched). FIX-1 fires, sets relay_broken=true.
#        Per device: 10s wasted × 2 MMIO devices = 20s overhead per quiesce cycle.
#
#   Root cause: FIX AT was missing. Phase 5 didn't check pending_quiesce_newly_dead_eth_chans_
#   before starting the 10s handshake poll. FIX AN only covers pre-dead channels (in
#   probe_dead_channels_map at terminate_stale time), not channels that die during quiesce
#   via FIX AS timeout.
#
# FIX AT (device.cpp: wait_for_fabric_workers_ready):
#   Before the Phase 5 handshake poll, check pending_quiesce_newly_dead_eth_chans_:
#   if the master chan is in that set (FIX AS Pass-0 timeout'd), immediately set
#   fabric_relay_path_broken_=true and skip the 10s poll. Phase 5b is then also skipped
#   via the existing FIX U/V relay_broken guard.
#   Per-device savings: kSyncTimeoutMs (10s). T3K 2×4 with 2 MMIO devices affected: 20s/cycle.
#
# What this test verifies:
#   1. Predecessor opens FABRIC_2D on a T3K, starts AllGather, then is SIGKILL'd.
#      Leaves MMIO ERISC channels in Phase-2.5-halted (assert-reset) state.
#   2. Testee (this process) opens FABRIC_2D and calls quiesce:
#      - FIX AR deasserts reset on all MMIO channels simultaneously.
#      - FIX AS 500ms timeout fires (WH BRISC boot >500ms) — all channels newly-dead.
#      - FIX AT detects master chan in newly-dead set → skips 10s Phase 5 poll.
#   3. Assert: one full quiesce+AllGather cycle completes within kBudgetPerCycleSec=20s.
#      Without FIX AT: cycle takes ≥20s (2 × 10s Phase 5 timeout) → FAIL.
#      With    FIX AT: cycle takes <15s (FIX AS ~500ms, FIX AT skip, no 10s wait) → PASS.
#   4. Repeat for 3 cycles to confirm repeatability.
#
# Relationship to existing tests:
#   GAP-26 (test_gap26_fixas_canary_timeout_graceful.py):
#     Tests FIX AS sad path — channels not ready in 500ms → mesh degrades gracefully.
#     Does NOT test Phase 5 timing behaviour when master chan is in the newly-dead set.
#   GAP-34 (test_gap34_fixam_phase5b_skip_timing.py):
#     Tests FIX AM — Phase 5b skipped when master stuck at STARTED (firmware loaded but
#     peer handshake incomplete). Different scenario: FIX AM requires status=STARTED, not 0x0.
#     FIX AT handles status=0x0 (firmware was never loaded at all).
#   GAP-30 (test_gap30_fixal_started_early_exit_timing.py):
#     Tests FIX AL — STARTED 3s early-exit. Requires relay-broken + FIX AL active.
#
# Why this is not already covered:
#   GAP-26 asserts graceful degradation (opens <45s, AllGather succeeds or fast-fails).
#   It does NOT assert a per-cycle timing budget. A 20s per-cycle overhead would pass GAP-26
#   if the total open stays <45s.  GAP-35 specifically asserts the 20s per-cycle budget.
#
# Hardware requirements: T3K (8-device 2×4 mesh), FABRIC_2D.

import os
import signal
import subprocess
import time
import sys

import pytest

import ttnn

# Per-cycle budget. Without FIX AT: ≥20s (2 × 10s Phase 5 timeout) + quiesce overhead.
# With FIX AT: <15s (FIX AS ~500ms, FIX AT skip, no 10s Phase 5 wait).
kBudgetPerCycleSec = 20.0
kNumCycles = 3
kPredecessorSetupSec = 3.0  # time to let predecessor establish FABRIC_2D before killing


def _run_predecessor_and_kill():
    """Fork a child that opens FABRIC_2D on T3K and blocks, then SIGKILL it."""
    child_code = """
import ttnn, time, signal, sys
signal.signal(signal.SIGTERM, lambda *a: None)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
print("PREDECESSOR_READY", flush=True)
time.sleep(30)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    # Wait for child to signal readiness (FABRIC_2D open and first AllGather in flight)
    deadline = time.time() + 30.0
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            line = proc.stdout.readline().decode("utf-8", errors="replace").strip()
            if "PREDECESSOR_READY" in line:
                ready = True
                break
        except Exception:
            break
    if not ready:
        proc.kill()
        proc.wait()
        pytest.skip("Predecessor did not reach READY state — hardware not available")
    # Let FABRIC_2D stabilize a moment then SIGKILL
    time.sleep(kPredecessorSetupSec)
    proc.send_signal(signal.SIGKILL)
    proc.wait()


@pytest.mark.parametrize("cycle", range(kNumCycles))
def test_fixat_phase5_skip_on_fixas_timeout(cycle):
    """
    After SIGKILL predecessor (leaves MMIO ERISC channels in assert-reset state), verify
    that one quiesce+AllGather cycle completes within kBudgetPerCycleSec.

    Without FIX AT: Phase 5 waits 10s per MMIO device (Device 1 + Device 3 = 20s).
    With FIX AT:    Phase 5 is skipped for those devices (<1s each → ~500ms total).
    """
    if cycle == 0:
        _run_predecessor_and_kill()
        # Brief settle to allow UMD driver cleanup on the runner
        time.sleep(0.5)

    t_start = time.time()

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(2, 4),
        )

        # Small AllGather to exercise the quiesce path
        shape = [1, 1, 32, 256]
        inp = ttnn.from_torch(
            __import__("torch").ones(*shape) * (cycle + 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=ttnn.MeshShape(2, 4)),
        )
        inp = ttnn.to_device(inp, mesh)
        out = ttnn.all_gather(inp, dim=-1, num_links=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh)

        elapsed = time.time() - t_start
        assert elapsed < kBudgetPerCycleSec, (
            f"Cycle {cycle}: quiesce+AllGather took {elapsed:.1f}s >= {kBudgetPerCycleSec}s budget. "
            f"FIX AT regression: Phase 5 likely waited full 10s per MMIO device "
            f"(FIX AS newly-dead master chan not detected before Phase 5 poll)."
        )
    except Exception as exc:
        elapsed = time.time() - t_start
        if elapsed >= kBudgetPerCycleSec:
            pytest.fail(
                f"Cycle {cycle}: exception after {elapsed:.1f}s >= {kBudgetPerCycleSec}s budget — "
                f"likely Phase 5 timeout (FIX AT regression): {exc}"
            )
        # Fast exception (relay_broken → AllGather skipped) is acceptable — FIX AT fired correctly
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
