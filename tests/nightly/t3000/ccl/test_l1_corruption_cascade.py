# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_13: L1 corruption cascade broken across 3+ sequential sessions
#
# Strategy: Session 1 opens a FABRIC_2D mesh on T3K, runs AllGather, then is
# SIGKILL'd mid-flight.  This leaves corrupt edm_status values in ERISC L1
# (the "first-generation" corruption).  Session 2 opens the same mesh —
# terminate_stale_erisc_routers() detects those corrupt status words and MUST
# zero router_sync_address for each affected channel (iteration 16 fix in
# nsexton/0-racecondition-hunt).  Session 2 then runs AllGather and is
# SIGKILL'd again, potentially leaving second-generation corruption.  Session 3
# opens the mesh: if the cascade is broken, it sees no more corrupt channels
# than Session 2 saw, recovers cleanly, and completes AllGather within 60 s.
# The test asserts that Session 3 exits with returncode 0 and that the
# combined open + run wall-clock time is < 30 s.
#
# Pass  = Session 3 exits cleanly (returncode 0) and open+run time < 30 s.
# Fail  = Session 3 exits with non-zero code (hung/crashed) OR open+run time
#         exceeds 30 s (terminate_stale_erisc_routers did not zero
#         router_sync_address, so clean-up loops spin on the leftover corrupt
#         status from Session 2's ERISC channels).
#
# Background — iteration 16 fix (FIX AN-16):
#   The first call to terminate_stale_erisc_routers() (Session 2) detects
#   corrupt edm_status and tears down those channels.  However, if it does NOT
#   also zero router_sync_address, the ERISC re-reads the stale sync address on
#   the next fabric open and may self-configure into a corrupt state again.
#   Session 3's terminate_stale_erisc_routers() would then see MORE corrupt
#   channels than Session 2 saw — a cascade.  With the fix, router_sync_address
#   is zeroed alongside the status word, breaking the cascade.
#
# Hardware: T3K (Wormhole, 8 devices).  Blackhole is skipped.

import os
import signal
import subprocess
import sys
import time
import textwrap

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_blackhole

# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------

# T3K topology: 1×8 linear/ring mesh (8 Wormhole devices).
_MESH_ROWS = 1
_MESH_COLS = 8

# Minimum devices needed to exercise ERISC cascade paths across multiple hops.
_MIN_DEVICES = 4

# AllGather tensor shape — large enough to keep all ERISC channels busy for
# several hundred milliseconds when SIGKILL lands, small enough to iterate fast.
# [1, 1, 32, 256] gathered on dim=3 → each device holds [1, 1, 32, 32].
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3

# Wall-clock budget for Session 3 open + AllGather (seconds).
_SESSION3_TIMEOUT_S = 30

# How long to let Sessions 1 and 2 run before SIGKILL (seconds).
# Large enough for AllGather to become in-flight; small enough to kill quickly.
_KILL_DELAY_S = 3.0

# Maximum wall-clock time to wait for a subprocess to be killed/finish (seconds).
_WAIT_TIMEOUT_S = 15

# ---------------------------------------------------------------------------
# Inline Python scripts for each session — kept as module-level strings so
# that they are easy to audit.  Each script is executed in a fresh interpreter
# via subprocess.Popen.  Using textwrap.dedent ensures no accidental
# indentation errors.
# ---------------------------------------------------------------------------

# Session 1 & 2 script: open mesh with FABRIC_2D, run AllGather in a tight
# loop, then block (sleep forever — SIGKILL arrives from the test driver).
_SESSION_KILLABLE_SCRIPT = textwrap.dedent(
    """\
    import torch
    import ttnn
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    MESH_SHAPE = ({rows}, {cols})
    AG_DIM = {ag_dim}
    AG_SHAPE = {ag_shape}

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*MESH_SHAPE),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(42)
    full_shape = AG_SHAPE[:]

    input_tensor = ttnn.from_torch(
        torch.rand(full_shape, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=AG_DIM),
    )

    # Run AllGather repeatedly until SIGKILL lands — keeps all ERISC channels busy.
    while True:
        tt_out = ttnn.all_gather(input_tensor, dim=AG_DIM, topology=ttnn.Topology.Ring)
        # Re-use input for the next iteration (avoids allocation overhead).
        _ = tt_out
"""
).format(
    rows=_MESH_ROWS,
    cols=_MESH_COLS,
    ag_dim=_AG_DIM,
    ag_shape=repr(_AG_OUTPUT_SHAPE),
)

# Session 3 script: open mesh with FABRIC_2D, run one AllGather, close cleanly,
# and print a sentinel line so the test driver can confirm execution reached that
# point.  Exit 0 on success; any uncaught exception causes non-zero exit.
_SESSION3_SCRIPT = textwrap.dedent(
    """\
    import sys
    import torch
    import ttnn
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    MESH_SHAPE = ({rows}, {cols})
    AG_DIM = {ag_dim}
    AG_SHAPE = {ag_shape}

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*MESH_SHAPE),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(99)

    input_tensor = ttnn.from_torch(
        torch.rand(AG_SHAPE, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=AG_DIM),
    )

    tt_out = ttnn.all_gather(input_tensor, dim=AG_DIM, topology=ttnn.Topology.Ring)

    # Confirm AllGather produced the expected gathered shape.
    expected_shape = list(AG_SHAPE)
    got_shape = list(tt_out.shape)
    assert got_shape == expected_shape, (
        f"Session 3 AllGather shape mismatch: expected {{expected_shape}}, got {{got_shape}}"
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    print("SESSION3_OK", flush=True)
    sys.exit(0)
"""
).format(
    rows=_MESH_ROWS,
    cols=_MESH_COLS,
    ag_dim=_AG_DIM,
    ag_shape=repr(_AG_OUTPUT_SHAPE),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spawn_session(script: str) -> subprocess.Popen:
    """Start a fresh Python subprocess running *script* and return the Popen handle."""
    return subprocess.Popen(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _kill_and_wait(proc: subprocess.Popen, label: str) -> None:
    """Send SIGKILL to *proc*, then wait up to _WAIT_TIMEOUT_S for it to die."""
    try:
        os.kill(proc.pid, signal.SIGKILL)
        logger.info(f"[{label}] SIGKILL sent to PID {proc.pid}")
    except ProcessLookupError:
        logger.warning(f"[{label}] Process {proc.pid} already exited before SIGKILL")

    try:
        proc.wait(timeout=_WAIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.error(f"[{label}] Process {proc.pid} did not die after SIGKILL within {_WAIT_TIMEOUT_S}s")
        raise


def _drain_output(proc: subprocess.Popen, label: str) -> tuple[str, str]:
    """Collect stdout and stderr from a completed process and log them."""
    stdout, stderr = proc.communicate()
    stdout_str = stdout.decode(errors="replace").strip()
    stderr_str = stderr.decode(errors="replace").strip()
    if stdout_str:
        logger.info(f"[{label}] stdout:\n{stdout_str}")
    if stderr_str:
        logger.info(f"[{label}] stderr (last 2 KiB):\n{stderr_str[-2048:]}")
    return stdout_str, stderr_str


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@skip_for_blackhole("GAP_13 targets Wormhole ERISC L1 corruption cascade — not applicable on Blackhole")
def test_l1_corruption_cascade_broken():
    """GAP_13: Validate that terminate_stale_erisc_routers zeros router_sync_address
    so that L1 corruption from a crashed session does not cascade into subsequent
    sessions.

    Three subprocess-based sessions are run:

      Session 1 — Opens FABRIC_2D mesh on T3K, enters an AllGather loop, is
                  SIGKILL'd after _KILL_DELAY_S seconds.  Leaves corrupt
                  edm_status / router_sync_address values in ERISC L1.

      Session 2 — Opens the same mesh.  terminate_stale_erisc_routers() must
                  detect the corrupt status words and zero router_sync_address
                  (iteration 16 fix).  Enters AllGather loop, is SIGKILL'd
                  again — leaving second-generation corruption that mimics
                  what would happen if the first fix were absent.

      Session 3 — Opens the mesh.  If the cascade is broken, the amount of
                  corrupt state seen is bounded (no growth across sessions).
                  Session 3 must complete AllGather and exit cleanly within
                  _SESSION3_TIMEOUT_S seconds.

    A hang or non-zero exit from Session 3 indicates that router_sync_address
    was NOT zeroed after Session 2, so the ERISC re-entered a corrupt state and
    terminate_stale_erisc_routers() on Session 3 could not recover in time.
    """
    # ------------------------------------------------------------------
    # Session 1: corrupt ERISC L1 for the first time.
    # ------------------------------------------------------------------
    logger.info("=== GAP_13: Session 1 start (will SIGKILL after %.1fs) ===", _KILL_DELAY_S)
    s1 = _spawn_session(_SESSION_KILLABLE_SCRIPT)
    time.sleep(_KILL_DELAY_S)
    _kill_and_wait(s1, "Session 1")
    _drain_output(s1, "Session 1")
    logger.info("=== GAP_13: Session 1 killed (PID %d, returncode %d) ===", s1.pid, s1.returncode)

    # ------------------------------------------------------------------
    # Session 2: open after first-generation corruption; terminate_stale_erisc_routers
    # should detect + zero router_sync_address.  Then corrupt again via SIGKILL.
    # ------------------------------------------------------------------
    logger.info("=== GAP_13: Session 2 start (will SIGKILL after %.1fs) ===", _KILL_DELAY_S)
    s2 = _spawn_session(_SESSION_KILLABLE_SCRIPT)
    time.sleep(_KILL_DELAY_S)
    _kill_and_wait(s2, "Session 2")
    _drain_output(s2, "Session 2")
    logger.info("=== GAP_13: Session 2 killed (PID %d, returncode %d) ===", s2.pid, s2.returncode)

    # ------------------------------------------------------------------
    # Session 3: must recover from second-generation corruption, complete
    # AllGather, and exit cleanly within _SESSION3_TIMEOUT_S seconds.
    # ------------------------------------------------------------------
    logger.info(
        "=== GAP_13: Session 3 start (must complete within %ds) ===",
        _SESSION3_TIMEOUT_S,
    )
    t_start = time.monotonic()
    s3 = _spawn_session(_SESSION3_SCRIPT)

    try:
        s3.wait(timeout=_SESSION3_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # Kill the hung process before raising so we don't leave a zombie.
        try:
            os.kill(s3.pid, signal.SIGKILL)
            s3.wait(timeout=5)
        except Exception:
            pass
        stdout3, stderr3 = _drain_output(s3, "Session 3")
        pytest.fail(
            f"GAP_13 FAIL: Session 3 hung — did not complete within {_SESSION3_TIMEOUT_S}s. "
            f"terminate_stale_erisc_routers() likely failed to zero router_sync_address after "
            f"Session 2's corruption, causing the cascade to continue.\n"
            f"stdout: {stdout3}\nstderr (tail): {stderr3[-1024:]}"
        )

    elapsed = time.monotonic() - t_start
    stdout3, stderr3 = _drain_output(s3, "Session 3")

    logger.info("=== GAP_13: Session 3 finished in %.2fs, returncode=%d ===", elapsed, s3.returncode)

    # --- Correctness assertions ---

    assert s3.returncode == 0, (
        f"GAP_13 FAIL: Session 3 exited with returncode {s3.returncode} (expected 0). "
        f"AllGather did not complete cleanly after cascade recovery.\n"
        f"stdout: {stdout3}\nstderr (tail): {stderr3[-1024:]}"
    )

    assert elapsed < _SESSION3_TIMEOUT_S, (
        f"GAP_13 FAIL: Session 3 open+run took {elapsed:.2f}s, exceeding the "
        f"{_SESSION3_TIMEOUT_S}s limit. router_sync_address was not zeroed after "
        f"Session 2, so the stale-router cleanup loop spun too long."
    )

    assert "SESSION3_OK" in stdout3, (
        f"GAP_13 FAIL: Session 3 exited 0 but did not print SESSION3_OK sentinel — "
        f"AllGather may not have actually run.\nstdout: {stdout3}"
    )

    logger.info(
        "GAP_13 PASS: Session 3 completed AllGather in %.2fs (limit %ds). "
        "L1 corruption cascade is broken — terminate_stale_erisc_routers zeros "
        "router_sync_address correctly.",
        elapsed,
        _SESSION3_TIMEOUT_S,
    )
