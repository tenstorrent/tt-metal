# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-84: OPTION C relay probe must not leave surviving relays in a marginal state
#
# Background — the concern:
#   OPTION C (fabric_firmware_initializer.cpp, teardown()) probes each surviving
#   non-MMIO relay device with a `detail::ReadFromDeviceL1` call on the master
#   ETH core's edm_status_address before calling l1_barrier to drain the UMD
#   relay queue.  The question: could that probe read — which goes through the
#   live UMD relay chain — leave the relay in a marginal state (partially-consumed
#   slot, in-flight transaction, disrupted link-layer state) that causes the
#   immediately following fabric init to fail or hang?
#
# What "marginal state" looks like in practice:
#   - Canary poll times out for a probed device's ETH channel (FIX AS / Pass-0)
#   - Ring-sync wait times out for a probed device's master router (FIX TK / FIX BZ)
#   - AllGather hangs or produces wrong output on a probed device's dimension
#   - write_launch_msg_to_core drops silently (FIX M path), appearing as zombie ERISC
#
# The OPTION C probe fires when:
#   `relay_dead_devices` (populated from `dev->is_fabric_relay_path_broken()` at
#   teardown entry) is non-empty AND the current device is NOT in relay_dead_devices.
#   Concretely: some non-MMIO devices had ring-sync timeout in the current session
#   (fabric_relay_path_broken_=true via FIX DY / FIX DX2), while other non-MMIO
#   devices completed ring sync successfully.
#
# Test scenario (3 sessions):
#
#   Session K (Kill): Opens T3K 2×4 mesh, runs one AllGather to activate all ERISC
#     channels, then is SIGKILL'd.  This leaves all ERISC channels in active FABRIC
#     firmware state — the prerequisite for FIX DY to detect partial ring-sync
#     failure in Session P.
#
#   Session P (Probe): Opens T3K mesh immediately after Session K.
#     - Phase 0: force-reset MMIO channels → reloaded with UMD relay
#     - FIX M: write_launch_msg_to_core to non-MMIO channels
#     - FIX DY: polls each non-MMIO ERISC for base-UMD → STARTED transition.
#       On a T3K where Session K left stale FABRIC state on some channels, FIX DY
#       may timeout on those channels → fabric_relay_path_broken_ = true.
#     - FIX DX2: dynamic re-check; partial ring-sync timeout possible.
#     - Fabric opens successfully on surviving links (or degrades gracefully).
#     - AllGather runs on the reduced mesh.
#     - Teardown:
#       a. If any device has relay_path_broken → relay_dead_devices non-empty.
#       b. For the surviving non-MMIO devices: OPTION C fires ReadFromDeviceL1 on
#          master ETH core, then l1_barrier.
#       c. This is the probe that must NOT leave marginal state.
#
#   Session V (Verify): Opens T3K mesh immediately after Session P closes.
#     - Must complete open within 45s (no ring-sync hang from marginal relay state).
#     - AllGather must succeed and produce correct output.
#     - Must NOT log any ring-sync timeout markers (FIX TK, FIX BZ, FIX DX2 bailout).
#
# Expected results:
#   1. Session P completes close within SESSION_P_BUDGET_S (≤ 90s).
#   2. Session V opens within SESSION_V_OPEN_BUDGET_S (≤ 45s).
#   3. Session V AllGather produces correct output (matches reference value).
#   4. Session V logs contain no ring-sync timeout markers.
#   5. If OPTION C probe DID fire in Session P, session V still passes (the main assertion).
#
# Failure mode caught:
#   If OPTION C's ReadFromDeviceL1 probe on a surviving relay leaves a partially-
#   consumed UMD relay queue slot or disrupts link state, Session V's
#   write_launch_msg_to_core (FIX M path) for that device will drop → FIX DY timeout
#   → ring-sync timeout → AllGather hang or skip.
#   This would show as SESSION_V_OPEN_ELAPSED > 45s or an AllGather assertion failure.
#
# Runs first in BATCH 2 (BATCH 2/2 in racecondition-hunt suite) so that a probe-induced
# marginal state is detected before any other ETH/ERISC regression test can mask it.
#
# Hardware: T3K (8 WH devices, 2×4 mesh). Requires >= 4 devices.

import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]  # AllGather output over 8 devices on dim=3
_AG_INPUT_SHAPE = [1, 1, 32, 32]    # Per-device shard

# Time budgets
_SESSION_K_KILL_AFTER_S = 3      # Kill Session K this many seconds after AllGather completes
_SESSION_P_BUDGET_S = 90          # Session P (probe) must close within this budget
_SESSION_V_OPEN_BUDGET_S = 45     # Session V open must complete within this budget
_SESSION_V_TOTAL_BUDGET_S = 120   # Total for Session V (open + allgather + close)
_BETWEEN_SESSION_DELAY_S = 1      # Delay between K→P and P→V to avoid link-training races

# Log markers that indicate ring-sync timeout / marginal state in Session V
_RING_SYNC_TIMEOUT_MARKERS = [
    "ring_sync_already_timed_out",
    "FIX TK",
    "FIX BZ",
    "fabric_ring_sync_timed_out",
    "Timeout after",
    "rescue of stuck dispatch cores",
    "OPTION C relay probe threw",           # probe failed in Session V (should not happen)
]

# Marker for OPTION C probe having fired (informational — not required to pass)
_OPTION_C_PROBE_FIRED_MARKER = "OPTION C (#42429): probing remaining non-MMIO"

# Session K: open T3K, run AllGather, then run indefinitely so parent can SIGKILL it.
_SESSION_K_SRC = textwrap.dedent("""
import sys, time, os
import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

ready_path = sys.argv[1]

mesh = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(2, 4),
    mesh_type=ttnn.MeshType.Ring,
)
ttnn.SetDefaultDevice(mesh)

t_in = ttnn.from_torch(
    torch.ones(1, 1, 32, 32, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
    mesh_mapper=ShardTensorToMesh(mesh, dim=3),
)
t_out = ttnn.all_gather(t_in, dim=3, num_links=1)
ttnn.deallocate(t_out)
ttnn.deallocate(t_in)
print("[GAP-84/K] AllGather done — all ERISC fabric FW active", flush=True)

# Signal parent that AllGather completed (all ERISCs active).
open(ready_path, "w").close()

# Run until SIGKILL — leaves all ERISC channels in active FABRIC state.
while True:
    time.sleep(0.1)
""")

# Session P: open T3K after SIGKILL'd predecessor → OPTION C probe should fire →
# AllGather → close.  Reports whether OPTION C fired and timing of close.
_SESSION_P_SRC = textwrap.dedent("""
import sys, time, re
import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
import logging

log_path = sys.argv[1]
logs = []

class LogCapture(logging.Handler):
    def emit(self, record):
        logs.append(self.format(record))

handler = LogCapture()
handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(handler)

open_start = time.time()
try:
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 4),
        mesh_type=ttnn.MeshType.Ring,
    )
except Exception as e:
    print(f"[GAP-84/P] open_mesh_device FAILED: {e}", flush=True)
    sys.exit(1)

open_elapsed = time.time() - open_start
print(f"[GAP-84/P] SESSION_P_OPEN_ELAPSED={open_elapsed:.2f}", flush=True)

ttnn.SetDefaultDevice(mesh)

try:
    t_in = ttnn.from_torch(
        torch.ones(1, 1, 32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )
    t_out = ttnn.all_gather(t_in, dim=3, num_links=1)
    ttnn.deallocate(t_out)
    ttnn.deallocate(t_in)
    print("[GAP-84/P] AllGather done", flush=True)
except Exception as e:
    print(f"[GAP-84/P] AllGather failed (acceptable — partial mesh): {e}", flush=True)

close_start = time.time()
ttnn.close_mesh_device(mesh)
close_elapsed = time.time() - close_start
print(f"[GAP-84/P] SESSION_P_CLOSE_ELAPSED={close_elapsed:.2f}", flush=True)

# Scan logs for OPTION C probe marker.
captured = "\\n".join(logs)
with open(log_path, "w") as f:
    f.write(captured)

sys.exit(0)
""")

# Session V: verify no marginal state — open + AllGather + close must all succeed.
_SESSION_V_SRC = textwrap.dedent("""
import sys, time
import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

open_start = time.time()
try:
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 4),
        mesh_type=ttnn.MeshType.Ring,
    )
except Exception as e:
    print(f"[GAP-84/V] open_mesh_device FAILED: {e}", flush=True, file=sys.stderr)
    sys.exit(1)

open_elapsed = time.time() - open_start
print(f"SESSION_V_OPEN_ELAPSED={open_elapsed:.2f}", file=sys.stderr, flush=True)

ttnn.SetDefaultDevice(mesh)

t_in = ttnn.from_torch(
    torch.ones(1, 1, 32, 32, dtype=torch.bfloat16) * 2.0,
    layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
    mesh_mapper=ShardTensorToMesh(mesh, dim=3),
)
t_out = ttnn.all_gather(t_in, dim=3, num_links=1)
t_torch = ttnn.to_torch(t_out, mesh_composer=ConcatMeshToTensor(mesh, dim=3))
ttnn.deallocate(t_out)
ttnn.deallocate(t_in)

import torch as _torch
expected = _torch.ones(1, 1, 32, 256, dtype=_torch.bfloat16) * 2.0
ok = _torch.allclose(t_torch.float(), expected.float())
print(f"ALLGATHER_CORRECT={ok}", file=sys.stderr, flush=True)
if not ok:
    print(f"[GAP-84/V] AllGather wrong: max_diff={(_torch.abs(t_torch.float() - expected.float())).max().item()}", file=sys.stderr, flush=True)
    sys.exit(2)

ttnn.close_mesh_device(mesh)
print("[GAP-84/V] close done", file=sys.stderr, flush=True)
sys.exit(0)
""")


def _is_fabric_degraded() -> bool:
    """Quick check: can we open a T3K 2x4 mesh?  Returns True if fabric is degraded."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import ttnn; m = ttnn.open_mesh_device(ttnn.MeshShape(2,4)); ttnn.close_mesh_device(m)"],
            timeout=45,
            capture_output=True,
        )
        return result.returncode != 0
    except subprocess.TimeoutExpired:
        return True


@pytest.mark.skipif(skip_for_blackhole(), reason="T3K (Wormhole) only")
def test_gap84_optc_probe_no_marginal_state():
    """
    OPTION C ReadFromDeviceL1 probe on surviving relays must not leave
    any surviving non-MMIO relay in a marginal state that prevents the
    immediately following fabric init from succeeding.

    Three sessions: K (SIGKILL'd, leaves stale FABRIC state) →
    P (probe session, OPTION C may fire) → V (verify, AllGather must work).
    """
    if _is_fabric_degraded():
        pytest.skip("Cluster fabric degraded before test start (prior test left stale state)")

    # -----------------------------------------------------------------------
    # Session K — open mesh, AllGather, SIGKILL
    # -----------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".ready", delete=False) as f:
        ready_path = f.name
    Path(ready_path).unlink(missing_ok=True)

    logger.info("[GAP-84] Starting Session K (AllGather + SIGKILL)")
    proc_k = subprocess.Popen(
        [sys.executable, "-c", _SESSION_K_SRC, ready_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for AllGather to complete (all ERISCs active)
    deadline = time.time() + 60
    while not Path(ready_path).exists():
        if time.time() > deadline:
            proc_k.kill()
            proc_k.wait()
            pytest.fail("Session K AllGather did not complete within 60s")
        time.sleep(0.2)

    logger.info("[GAP-84] Session K AllGather confirmed — sleeping then SIGKILL")
    time.sleep(_SESSION_K_KILL_AFTER_S)
    proc_k.send_signal(signal.SIGKILL)
    proc_k.wait(timeout=5)
    logger.info("[GAP-84] Session K SIGKILL'd (all ERISC channels in stale FABRIC state)")
    Path(ready_path).unlink(missing_ok=True)

    time.sleep(_BETWEEN_SESSION_DELAY_S)

    # -----------------------------------------------------------------------
    # Session P — open mesh (may trigger OPTION C probe), AllGather, close
    # -----------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as f:
        p_log_path = f.name

    logger.info("[GAP-84] Starting Session P (OPTION C probe session)")
    p_start = time.time()
    result_p = subprocess.run(
        [sys.executable, "-c", _SESSION_P_SRC, p_log_path],
        timeout=_SESSION_P_BUDGET_S + 30,
        capture_output=True,
    )
    p_elapsed = time.time() - p_start
    p_stdout = result_p.stdout.decode(errors="replace")
    p_stderr = result_p.stderr.decode(errors="replace")

    logger.info(f"[GAP-84] Session P completed in {p_elapsed:.1f}s (rc={result_p.returncode})")
    logger.debug(f"[GAP-84] Session P stdout:\n{p_stdout[-1000:]}")

    # Read captured Metal logs from Session P
    try:
        p_logs = Path(p_log_path).read_text()
    except Exception:
        p_logs = ""

    # Check if OPTION C probe actually fired (informational — test passes either way)
    option_c_fired = _OPTION_C_PROBE_FIRED_MARKER in p_stdout or _OPTION_C_PROBE_FIRED_MARKER in p_logs
    if option_c_fired:
        logger.info("[GAP-84] OPTION C relay probe DID fire in Session P (probe path exercised)")
    else:
        logger.info(
            "[GAP-84] OPTION C relay probe did NOT fire in Session P "
            "(all non-MMIO relays alive OR no dead-relay devices at teardown — "
            "test still validates marginal-state safety)"
        )

    # Extract Session P close elapsed from stdout
    p_close_elapsed = None
    for line in p_stdout.splitlines():
        m = re.search(r"SESSION_P_CLOSE_ELAPSED=(\d+\.\d+)", line)
        if m:
            p_close_elapsed = float(m.group(1))
    if p_close_elapsed is not None:
        logger.info(f"[GAP-84] Session P teardown (close) elapsed: {p_close_elapsed:.2f}s")
        assert p_close_elapsed < _SESSION_P_BUDGET_S, (
            f"Session P teardown took {p_close_elapsed:.1f}s — exceeds {_SESSION_P_BUDGET_S}s budget. "
            f"OPTION C probe may have hung on a surviving relay."
        )

    assert result_p.returncode == 0, (
        f"Session P exited with rc={result_p.returncode} — fabric open/close failed.\n"
        f"stdout: {p_stdout[-800:]}\nstderr: {p_stderr[-400:]}"
    )

    time.sleep(_BETWEEN_SESSION_DELAY_S)

    # -----------------------------------------------------------------------
    # Session V — verify: open + AllGather must succeed, no ring-sync hang
    # -----------------------------------------------------------------------
    logger.info("[GAP-84] Starting Session V (verify no marginal relay state)")
    v_start = time.time()
    result_v = subprocess.run(
        [sys.executable, "-c", _SESSION_V_SRC],
        timeout=_SESSION_V_TOTAL_BUDGET_S + 30,
        capture_output=True,
    )
    v_elapsed = time.time() - v_start
    v_stderr = result_v.stderr.decode(errors="replace")
    v_stdout = result_v.stdout.decode(errors="replace")

    logger.info(f"[GAP-84] Session V completed in {v_elapsed:.1f}s (rc={result_v.returncode})")

    # Extract Session V open elapsed
    v_open_elapsed = None
    for line in v_stderr.splitlines():
        m = re.search(r"SESSION_V_OPEN_ELAPSED=(\d+\.\d+)", line)
        if m:
            v_open_elapsed = float(m.group(1))

    if v_open_elapsed is not None:
        logger.info(f"[GAP-84] Session V open elapsed: {v_open_elapsed:.2f}s")
        assert v_open_elapsed < _SESSION_V_OPEN_BUDGET_S, (
            f"Session V open took {v_open_elapsed:.1f}s — exceeds {_SESSION_V_OPEN_BUDGET_S}s budget. "
            f"Suspected marginal relay state left by OPTION C probe in Session P. "
            f"Session P OPTION C fired: {option_c_fired}.\n"
            f"Session V stderr: {v_stderr[-600:]}"
        )
    else:
        # If we can't parse, bound by total elapsed
        assert v_elapsed < _SESSION_V_OPEN_BUDGET_S + 30, (
            f"Session V total elapsed {v_elapsed:.1f}s likely indicates open hang. "
            f"stderr: {v_stderr[-400:]}"
        )

    # Check for ring-sync timeout markers in Session V output
    combined_v = v_stdout + v_stderr
    for marker in _RING_SYNC_TIMEOUT_MARKERS:
        assert marker not in combined_v, (
            f"Session V logged ring-sync timeout marker '{marker}' — suspected marginal relay "
            f"state left by OPTION C probe in Session P (probe fired: {option_c_fired}).\n"
            f"Session V stderr: {v_stderr[-600:]}"
        )

    # AllGather correctness
    allgather_correct = "ALLGATHER_CORRECT=True" in combined_v
    assert result_v.returncode == 0 and allgather_correct, (
        f"Session V AllGather failed (rc={result_v.returncode}, correct={allgather_correct}). "
        f"Suspected marginal relay state from OPTION C probe in Session P "
        f"(probe fired: {option_c_fired}).\n"
        f"Session V stderr: {v_stderr[-600:]}"
    )

    logger.info(
        f"[GAP-84] PASS — Session V AllGather correct, open={v_open_elapsed:.1f}s, "
        f"OPTION C probe fired: {option_c_fired}"
    )
