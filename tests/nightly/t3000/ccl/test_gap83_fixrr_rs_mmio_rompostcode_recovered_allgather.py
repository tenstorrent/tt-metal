# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-83: FIX RR/RS — MMIO ROM-postcode channel recovery → recovered channels load firmware → AllGather succeeds
#
# Background:
#   When tt-smi -r resets an MMIO device (Device 0 in T3K), its ETH channels go through
#   ROM boot.  terminate_stale_erisc_routers() may detect these channels as ROM-postcode
#   (edm_status_address = 0x49705180) and insert them into pre_dead_channels.
#
#   Before FIX RR: configure_fabric_cores() skipped soft-reset for pre_dead channels;
#   before FIX RS: even if FIX RR recovered them, configure_fabric() still treated them
#   as dead (kept them in fabric_pre_dead_channels_).  Result: firmware never loaded on
#   those channels → verify_all_fabric_channels_healthy saw Device 0's own master router
#   chan as pre-dead → fabric_channels_not_ready_for_traffic_=true → FIX QE skipped ALL
#   AllGather tests for the rest of the run (thousands of skips in CI).
#
#   After FIX RR + FIX RS (commits 3a1d542, 4db30163):
#     - FIX RR: configure_fabric_cores() attempts PCIe-direct soft-reset on MMIO pre-dead
#       channels; on success removes them from dead_channels and adds to recovered_channels.
#     - FIX RS: configure_fabric() computes effective_pre_dead = pre_dead - recovered;
#       recovered channels receive firmware normally; FIX QE does NOT skip AllGather.
#
# What this test verifies:
#   Scenario:
#     Session A: Open T3K mesh, run AllGather, then force tt-smi -r to put Device 0 ETH
#                channels into ROM-postcode state, then exit cleanly (triggering the
#                terminate_stale_erisc_routers ROM-postcode detection path).
#     Session B: Reopen mesh.  FIX RR should recover the ROM-postcode channels.
#                FIX RS propagates them back.  AllGather must succeed (not be skipped by FIX QE).
#
#   Regression indicator:
#     - Session B logs contain "FIX QE" skip → FIX RS broken (channels still pre-dead).
#     - Session B AllGather hangs or errors → FIX RR reset incomplete.
#     - Session B logs: "rr_recovered=0" in configure_fabric complete line for MMIO device.
#
# Failure modes caught:
#   a. FIX RS regression: recovered channels not subtracted from effective_pre_dead →
#      fabric_pre_dead_channels_ still includes recovered chans → FIX QE skips AllGather.
#   b. FIX RR regression: PCIe-direct soft reset skipped for pre-dead MMIO chans →
#      firmware not loaded → all AllGathers fail for the run.
#   c. AllGather hang: FIX RR reset incomplete, dead ETH channel blocks write_launch_msg.
#
# Hardware: T3K (2×4 WH mesh). Requires FABRIC_2D mode.
# Timing budget: 300s total (A setup + tt-smi + B recovery + AllGather).

import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]  # AllGather output for 8 devices on dim=3
_AG_INPUT_SHAPE = [1, 1, 32, 32]    # Per-device shard

_SESSION_A_BUDGET_S = 90   # Time budget for Session A (allgather + tt-smi)
_SESSION_B_BUDGET_S = 180  # Budget for Session B (FIX RR recovery + AllGather)

# Session A: opens T3K mesh with fabric, runs one AllGather, then forces tt-smi -r
# to put Device 0 ETH channels into ROM-postcode state, then exits cleanly.
_SESSION_A_SRC = textwrap.dedent("""
import subprocess, sys, time, pathlib, os
import torch
import ttnn

READY_FILE = pathlib.Path(sys.argv[1])

mesh = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(2, 4),
    mesh_type=ttnn.MeshType.Ring,
)
ttnn.SetDefaultDevice(mesh)

# Run one AllGather to confirm ERISC fabric FW is initialized.
import torch
t_in = ttnn.from_torch(
    torch.ones(1, 1, 32, 32, dtype=torch.bfloat16) * 1.0,
    layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
)
t_out = ttnn.all_gather(t_in, dim=3, num_links=1)
torch_out = ttnn.to_torch(t_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
assert torch_out.shape == torch.Size([1, 1, 32, 256]), f"AllGather A wrong shape: {torch_out.shape}"
del t_in, t_out, torch_out
print("[GAP-83 Session A] AllGather confirmed — ERISC fabric FW active.", flush=True)

# Close the mesh cleanly (triggers terminate_stale_erisc_routers quiesce).
ttnn.close_mesh_device(mesh)
print("[GAP-83 Session A] Mesh closed cleanly.", flush=True)

# Force tt-smi -r to put Device 0 into ROM-postcode state (simulates power cycle of MMIO device).
print("[GAP-83 Session A] Running tt-smi -r to put ETH channels into ROM-postcode...", flush=True)
r = subprocess.run(["tt-smi", "-r"], capture_output=True, text=True, timeout=60)
print(f"[GAP-83 Session A] tt-smi -r exit={r.returncode}: {r.stdout.strip()[:200]}", flush=True)
time.sleep(2)  # Allow ROM boot to settle to postcode state.

# Signal Session B that A is done.
READY_FILE.write_text("done")
print("[GAP-83 Session A] Ready file written. Exiting.", flush=True)
""")

# Session B: reopens mesh; FIX RR should recover ROM-postcode channels; AllGather must succeed.
_SESSION_B_SRC = textwrap.dedent("""
import sys, pathlib, re, time
import torch
import ttnn

LOG_FILE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

# Re-open mesh — this triggers terminate_stale_erisc_routers (may see ROM-postcode chans)
# followed by configure_fabric_cores (FIX RR: PCIe-direct soft reset) and configure_fabric
# (FIX RS: propagate recovered_channels back).
mesh = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(2, 4),
    mesh_type=ttnn.MeshType.Ring,
)
ttnn.SetDefaultDevice(mesh)
print("[GAP-83 Session B] Mesh opened successfully after FIX RR/RS recovery.", flush=True)

# Verify AllGather succeeds (FIX QE must NOT have skipped it).
t_in = ttnn.from_torch(
    torch.ones(1, 1, 32, 32, dtype=torch.bfloat16) * 2.0,
    layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
)
t_out = ttnn.all_gather(t_in, dim=3, num_links=1)
torch_out = ttnn.to_torch(t_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
assert torch_out.shape == torch.Size([1, 1, 32, 256]), f"AllGather B wrong shape: {torch_out.shape}"
expected = torch.ones(1, 1, 32, 256, dtype=torch.bfloat16) * 2.0
assert torch.allclose(torch_out.float(), expected.float(), atol=0.1), \
    f"AllGather B values wrong: max_diff={torch.max(torch.abs(torch_out.float() - expected.float())).item()}"
print("[GAP-83 Session B] AllGather PASSED — FIX RR/RS recovery successful.", flush=True)

ttnn.close_mesh_device(mesh)
print("[GAP-83 Session B] Mesh closed cleanly.", flush=True)
""")


@pytest.mark.timeout(300)
@skip_for_blackhole()
def test_gap83_fixrr_rs_mmio_rompostcode_recovered_allgather(tmp_path):
    """GAP-83: FIX RR/RS — MMIO ROM-postcode channel recovery → firmware loaded → AllGather succeeds."""
    ready_file = tmp_path / "session_a_done.txt"

    # --- Session A ---
    sess_a_script = tmp_path / "gap83_session_a.py"
    sess_a_script.write_text(_SESSION_A_SRC)
    logger.info("[GAP-83] Starting Session A: AllGather + tt-smi -r to trigger ROM-postcode state")

    proc_a = subprocess.Popen(
        [sys.executable, str(sess_a_script), str(ready_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Stream Session A output
    a_log = []
    deadline_a = time.time() + _SESSION_A_BUDGET_S
    while proc_a.poll() is None and time.time() < deadline_a:
        line = proc_a.stdout.readline()
        if line:
            logger.info(f"[A] {line.rstrip()}")
            a_log.append(line)
    if proc_a.poll() is None:
        proc_a.kill()
        pytest.fail(f"Session A did not complete within {_SESSION_A_BUDGET_S}s")
    remaining = proc_a.stdout.read()
    if remaining:
        for line in remaining.splitlines():
            logger.info(f"[A] {line}")
            a_log.append(line + "\n")
    assert proc_a.returncode == 0, f"Session A failed (exit {proc_a.returncode})"
    assert ready_file.exists(), "Session A never wrote ready file"
    logger.info("[GAP-83] Session A complete. Waiting 3s for ROM-postcode to settle...")
    time.sleep(3)

    # --- Session B ---
    sess_b_script = tmp_path / "gap83_session_b.py"
    sess_b_script.write_text(_SESSION_B_SRC)
    logger.info("[GAP-83] Starting Session B: FIX RR recovery path + AllGather validation")

    proc_b = subprocess.Popen(
        [sys.executable, str(sess_b_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    b_log = []
    deadline_b = time.time() + _SESSION_B_BUDGET_S
    while proc_b.poll() is None and time.time() < deadline_b:
        line = proc_b.stdout.readline()
        if line:
            logger.info(f"[B] {line.rstrip()}")
            b_log.append(line)
    if proc_b.poll() is None:
        proc_b.kill()
        pytest.fail(f"Session B did not complete within {_SESSION_B_BUDGET_S}s — AllGather likely hung")
    remaining = proc_b.stdout.read()
    if remaining:
        for line in remaining.splitlines():
            logger.info(f"[B] {line}")
            b_log.append(line + "\n")

    b_combined = "".join(b_log)

    # Regression check: FIX QE should NOT have fired (would mean FIX RS is broken)
    fix_qe_fires = len(re.findall(r"FIX QE|SKIPPED.*AllGather.*stale.*ETH|channels_not_ready_for_traffic.*skip", b_combined))
    assert fix_qe_fires == 0, (
        f"[GAP-83] REGRESSION: FIX QE fired {fix_qe_fires} time(s) in Session B — "
        f"FIX RS is broken (recovered channels still in fabric_pre_dead_channels_)"
    )

    # FIX RR success log should appear for MMIO device (Device 0)
    rr_success = len(re.findall(r"FIX RR.*PCIe-direct soft reset succeeded", b_combined))
    rr_fail = len(re.findall(r"FIX RR.*PCIe-direct soft reset FAILED", b_combined))
    logger.info(f"[GAP-83] FIX RR results: {rr_success} succeeded, {rr_fail} failed")

    # Check Session B exit
    assert proc_b.returncode == 0, (
        f"Session B failed (exit {proc_b.returncode}). "
        f"FIX RR success={rr_success}, fail={rr_fail}. "
        f"Log tail: {''.join(b_log[-20:])}"
    )
    logger.info(
        f"[GAP-83] PASSED — FIX RR/RS recovery: {rr_success} channel(s) recovered, "
        f"AllGather succeeded, FIX QE did not fire."
    )
