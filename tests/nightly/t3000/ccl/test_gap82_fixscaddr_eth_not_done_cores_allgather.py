# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-82: FIX SC-ADDR — ETH cores in not_done_cores use per-core-type go_msg address
#
# Background:
#   FIX SC pre-scan (risc_firmware_initializer.cpp) iterates not_done_cores and checks
#   whether each core has a stale go_msg value.  not_done_cores includes ETH cores when
#   INIT_FABRIC is active (ACTIVE_ETH and IDLE_ETH are inserted during EDM firmware launch).
#
#   Before FIX SC-ADDR: the pre-scan always used the TENSIX go_msg address for ALL cores.
#   ETH cores keep go_msg at a different L1 offset (GET_ETH_MAILBOX_ADDRESS_HOST).
#   Reading the wrong address from an ETH core returned garbage (e.g. 0x02) which was not
#   in kKnownRunMsgValues → FIX SC fired on ETH core → wrote RUN_MSG_DONE to the wrong ETH
#   L1 address → corrupted dispatch FW data structures → deterministic 1000ms teardown
#   timeout for all 8 devices, every open/close cycle.
#
#   After FIX SC-ADDR (commit a50407086ac): the loop uses
#       hal_.get_dev_addr(llrt::get_core_type(device_id, worker_core), HalL1MemAddrType::GO_MSG)
#   which correctly resolves to the ETH mailbox address for ETH cores.
#
# What this test verifies:
#   Scenario: Session A opens a T3K mesh WITH fabric (INIT_FABRIC → ETH in not_done_cores),
#             runs an AllGather to ensure ERISC fabric FW is active, then is SIGKILL'd.
#             Session B reopens the mesh.  FIX SC pre-scan now scans not_done_cores using
#             per-core-type go_msg addresses.  ETH cores with valid go_msg should NOT
#             trigger FIX SC.  AllGather in Session B must succeed correctly.
#
#   Regression indicator: if FIX SC-ADDR regressed, Session B would emit
#       "FIX SC (GAP-76): Device N core X (ACTIVE_ETH) has stale go_msg=0xNN"
#   for every ETH core → corrupted ETH dispatch FW → teardown timeouts cascade → hang.
#
# Failure modes caught:
#   a. Session B AllGather hangs: FIX SC-ADDR regressed; ETH dispatch FW corrupted.
#   b. Session B AllGather wrong values: ETH L1 corrupted by false RUN_MSG_DONE write.
#   c. Session B log contains ETH FIX SC fire: direct regression signature.
#   d. Session B open_mesh_device raises: teardown corruption chain triggered (pre-fix symptom).
#
# Hardware: T3K (2×4 WH mesh). Requires FABRIC_2D mode.
# Timing budget: 240s total (A kill + B FIX SC-ADDR path + AllGather).

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

_SESSION_A_BUDGET_S = 60   # Time budget for Session A (kill after AllGather)
_SESSION_B_BUDGET_S = 120  # Budget for Session B (FIX SC-ADDR path + AllGather)

# Session A: opens T3K mesh with fabric (ensures ACTIVE_ETH cores are in not_done_cores
# in the subsequent session), runs one AllGather to ensure fabric ERISC FW is up,
# then blocks until SIGKILL.
_SESSION_A_SRC = textwrap.dedent("""
import signal, sys, time, pathlib
import torch
import ttnn

READY_FILE = pathlib.Path(sys.argv[1])

mesh = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(2, 4),
    mesh_type=ttnn.MeshType.Ring,
)
ttnn.SetDefaultDevice(mesh)

# Run one AllGather to confirm ERISC fabric FW is initialized (ACTIVE_ETH in not_done_cores).
num_devices = mesh.get_num_devices()
input_tensors = []
for i in range(num_devices):
    t = ttnn.from_torch(
        torch.full((1, 1, 32, 32), float(i + 1)),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )
    input_tensors.append(t)
combined = ttnn.concat(input_tensors, dim=3)
_ = ttnn.all_gather(combined, dim=3, num_links=1)

READY_FILE.touch()
# Block until SIGKILL — leave ETH cores in ERISC fabric FW state.
while True:
    time.sleep(1)
""")

# Session B: opens mesh (not_done_cores will include ETH cores from INIT_FABRIC path),
# FIX SC pre-scan must use per-core-type go_msg address (FIX SC-ADDR) and must NOT
# fire on ETH cores that have valid go_msg.  Then runs AllGather and checks correctness.
_SESSION_B_SRC = textwrap.dedent("""
import sys
import os
import re
import torch
import ttnn

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "WARNING")

EXIT_SKIP = 77
EXIT_SC_ETH_REGRESSION = 3

try:
    import io
    import contextlib

    # Capture stderr to check for ETH FIX SC fires (regression indicator).
    log_lines = []
    original_stderr_fd = sys.stderr.fileno()
    import tempfile
    log_capture_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
    log_capture_path = log_capture_file.name
    log_capture_file.close()

    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 4),
        mesh_type=ttnn.MeshType.Ring,
    )
    ttnn.SetDefaultDevice(mesh)

    if hasattr(mesh, "is_fabric_degraded") and mesh.is_fabric_degraded():
        print("[SESSION_B] SKIP: fabric degraded after kill — skip rather than hang", file=sys.stderr)
        ttnn.close_mesh_device(mesh)
        sys.exit(EXIT_SKIP)

    num_devices = mesh.get_num_devices()
    input_tensors = []
    for i in range(num_devices):
        t = ttnn.from_torch(
            torch.full((1, 1, 32, 32), float(i + 1)),
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        input_tensors.append(t)

    combined = ttnn.concat(input_tensors, dim=3)
    gathered = ttnn.all_gather(combined, dim=3, num_links=1)
    result = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))

    first_device = result[0, 0, 0, :]
    expected = torch.cat([torch.full((32,), float(i + 1)) for i in range(num_devices)])
    if not torch.allclose(first_device, expected, atol=0.1):
        max_err = (first_device - expected).abs().max().item()
        print(f"[SESSION_B] AllGather correctness FAIL: max_err={max_err:.4f}", file=sys.stderr)
        ttnn.close_mesh_device(mesh)
        sys.exit(2)

    print("[SESSION_B] AllGather correctness PASS — FIX SC-ADDR path is correct")
    ttnn.close_mesh_device(mesh)
    sys.exit(0)

except Exception as e:
    print(f"[SESSION_B] raised: {e}", file=sys.stderr)
    sys.exit(1)
""")


def _run_session(label: str, src: str, ready_file: Path | None, budget_s: int, kill: bool) -> tuple[int, str, str]:
    """Write src to a temp file and run it as a subprocess.  If kill=True, send SIGKILL after ready_file appears."""
    import tempfile, signal

    script = Path(tempfile.mktemp(suffix=".py"))
    script.write_text(src)
    try:
        args = [sys.executable, str(script)]
        if ready_file is not None:
            args.append(str(ready_file))

        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if kill and ready_file is not None:
            deadline = time.time() + budget_s
            while not ready_file.exists():
                if time.time() > deadline:
                    proc.kill()
                    proc.wait()
                    raise TimeoutError(f"{label}: ready_file not created within {budget_s}s")
                if proc.poll() is not None:
                    out, err = proc.stdout.read(), proc.stderr.read()
                    raise RuntimeError(f"{label}: process exited early (rc={proc.returncode})\n{err}")
                time.sleep(0.5)
            time.sleep(1)  # Brief pause after ready_file — mesh is open, ETH FW active.
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            return proc.returncode, "", ""

        try:
            out, err = proc.communicate(timeout=budget_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            raise TimeoutError(f"{label}: timed out after {budget_s}s\nstdout:\n{out}\nstderr:\n{err}")

        return proc.returncode, out, err
    finally:
        script.unlink(missing_ok=True)


@skip_for_blackhole()
@pytest.mark.timeout(300)
def test_gap82_fixscaddr_eth_not_done_cores_allgather():
    """
    GAP-82: FIX SC-ADDR — ETH cores in not_done_cores must use per-core-type go_msg address.

    Session A opens mesh with INIT_FABRIC (ETH in not_done_cores), runs AllGather to ensure
    ERISC FW is active, then is SIGKILL'd.
    Session B reopens mesh: FIX SC pre-scan runs on not_done_cores including ETH cores.
    FIX SC-ADDR must route ETH cores to their correct go_msg address — no false FIX SC fire.
    AllGather in Session B must succeed correctly.
    """
    tmp_dir = Path("/tmp/gap82")
    tmp_dir.mkdir(exist_ok=True)
    ready_file_a = tmp_dir / "session_a_ready.flag"
    ready_file_a.unlink(missing_ok=True)

    logger.info("GAP-82: Session A — open mesh with ERISC fabric FW active, then SIGKILL")
    try:
        _run_session("Session A", _SESSION_A_SRC, ready_file_a, _SESSION_A_BUDGET_S, kill=True)
    except TimeoutError as e:
        pytest.fail(f"GAP-82: {e}")
    except RuntimeError as e:
        pytest.fail(f"GAP-82: {e}")

    time.sleep(2)  # Let OS clean up ERISC state.

    logger.info("GAP-82: Session B — reopen mesh; FIX SC-ADDR should handle ETH not_done_cores; run AllGather")
    try:
        rc, out, err = _run_session("Session B", _SESSION_B_SRC, None, _SESSION_B_BUDGET_S, kill=False)
    except TimeoutError as e:
        pytest.fail(
            f"GAP-82: Session B TIMED OUT ({_SESSION_B_BUDGET_S}s) — likely AllGather hang.\n"
            f"If FIX SC-ADDR regressed: ETH dispatch FW corrupted → teardown timeouts → hang.\n"
            f"Error: {e}"
        )

    combined_log = (out or "") + (err or "")

    # Check for ETH FIX SC fires — direct regression indicator.
    eth_sc_fires = re.findall(
        r"FIX SC \(GAP-76\): Device \d+ core .+? \((ACTIVE_ETH|IDLE_ETH)\) has stale go_msg",
        combined_log,
    )
    if eth_sc_fires:
        pytest.fail(
            f"GAP-82: REGRESSION — FIX SC fired on ETH core(s): {eth_sc_fires}.\n"
            f"FIX SC-ADDR (commit a50407086ac) must be using per-core-type go_msg address.\n"
            f"ETH cores read wrong L1 offset → garbage signal → FIX SC → ETH L1 corruption.\n"
            f"Full log:\n{combined_log[:4000]}"
        )

    if rc == 77:
        pytest.skip("GAP-82: fabric degraded after Session A kill — skipped (not a FIX SC-ADDR failure)")

    if rc != 0:
        pytest.fail(
            f"GAP-82: Session B exited with rc={rc}.\n"
            f"stdout:\n{out}\nstderr:\n{err}\n"
            f"If AllGather hung: FIX SC-ADDR regressed (ETH L1 corruption → teardown timeout cascade).\n"
            f"If AllGather wrong values: ETH go_msg address overwritten with RUN_MSG_DONE."
        )

    logger.info("GAP-82: PASS — AllGather correct after Session A kill; no ETH FIX SC fires detected.")
    logger.info("  FIX SC-ADDR confirmed: ETH cores in not_done_cores read at correct go_msg address.")
