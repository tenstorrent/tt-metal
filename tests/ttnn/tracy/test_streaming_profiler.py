# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming profiler end-to-end test over the DRISC relay path.

Runs the ``test_streaming_profiler_zones`` workload with ``TT_METAL_STREAMING_PROFILER=1`` and
``TT_METAL_STREAMING_PROFILER_TRACY=1`` under a connected ``tracy-capture``, and checks that the relays go
resident at bring-up and that the capture holds device zones from every RISC of every workload core, read back with
``tracy-csvexport -u``. Needs a Blackhole box with DRAM programmable cores; device work runs in a subprocess so the
pytest parent never takes the PCIe lock.
"""

from __future__ import annotations

import csv
import io
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tools.tracy.common import PROFILER_ARTIFACTS_DIR, PROFILER_BIN_DIR, TT_METAL_HOME


def _workload_bin() -> Path:
    """CI builds into ``build`` (build-artifact.yaml passes ``--build-dir build``); a local build_metal.sh run
    leaves ``build`` as a symlink to ``build_Release``. Try both so one path works in either layout -- naming
    only ``build_Release`` made this test silently skip in CI."""
    rel = Path("programming_examples") / "test_streaming_profiler_zones"
    for d in ("build", "build_Release"):
        cand = Path(TT_METAL_HOME) / d / rel
        if cand.exists():
            return cand
    return Path(TT_METAL_HOME) / "build" / rel


WORKLOAD_BIN = _workload_bin()

# Every external program this module runs is named here, and commands are built from this table rather than
# assembled from data: an unknown key raises instead of executing anything.
TOOLS = {
    "capture": PROFILER_BIN_DIR / "tracy-capture",
    "csvexport": PROFILER_BIN_DIR / "tracy-csvexport",
    "workload": WORKLOAD_BIN,
}
RISCS_PER_CORE = 5
DM_RISCS_PER_CORE = 2  # only BRISC/NCRISC emit the point-marker trio
ZONES_PER_ITER = 10
MARKERS_PER_ITER = 3  # _Event (bare) + _Data and _Iter (timestamped payloads)
ARTIFACTS = PROFILER_ARTIFACTS_DIR / "streaming_profiler_tests"
CAPTURE_FILE = ARTIFACTS / "streaming_profiler_zones.tracy"

# Workload shape. These are the single source of truth: the parametrize cases below are built from them, so a
# command and the counts asserted against it can never drift apart.
GRID_X, GRID_Y = 2, 2
SMOKE_ITERS = 20
CAPTURE_ITERS = 50

# Complete commands, fixed at import. A caller picks one by key; no caller-supplied value is ever spliced into
# an argument list, which keeps the set of programs and arguments this module can run closed and reviewable.
COMMANDS = {
    "csvexport": [str(TOOLS["csvexport"]), "-u", str(CAPTURE_FILE)],
    "workload-smoke": [
        str(TOOLS["workload"]),
        "--gx",
        str(GRID_X),
        "--gy",
        str(GRID_Y),
        "--iters",
        str(SMOKE_ITERS),
        "--markers",
        "1",
    ],
    "workload-capture": [
        str(TOOLS["workload"]),
        "--gx",
        str(GRID_X),
        "--gy",
        str(GRID_Y),
        "--iters",
        str(CAPTURE_ITERS),
        "--markers",
        "1",
    ],
}


def _tool(name: str) -> str:
    """Resolve one of TOOLS. The name is always a literal at the call site, so the program being run is fixed
    by this module and can never come from a caller or the environment."""
    exe = TOOLS[name]
    assert exe.is_file(), f"{name} not built at {exe}"
    return str(exe)


def _free_port() -> str:
    ip = socket.gethostbyname(socket.gethostname())
    for port in range(8086, 8500):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, port))
            s.close()
            return str(port)
        except (PermissionError, OSError):
            continue
    raise RuntimeError("no free TCP port for tracy-capture")


def _run_tool(which: str, *, env: dict | None = None, cwd: str | None = None, timeout: int) -> tuple[int, str, str]:
    """Run the COMMANDS entry named by ``which`` and wait for it: returns (returncode, stdout, stderr).

    Callers name a command rather than supplying one, so the argument list always comes from the table above
    and an unknown name raises instead of running anything. Single place a child process is created: no shell,
    and a hung tool is killed rather than left to block the suite. (The Tracy capture is the one exception --
    it has to run alongside the workload, so it is started directly and reaped in a finally.)
    """
    argv = COMMANDS[which]
    proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return proc.returncode, out, err


def _device_zone_rows() -> list[dict]:
    """Every device-zone row in the capture. The Tracy sink stamps device zones with the source file
    ``kernel_profiler``, which is what separates them from host zones in the export. One export answers both the
    lane count and the name set; exporting the same capture twice was pure duplication.

    The argument list is fixed and passed without a shell (shell=False), so nothing here is parsed as a command:
    the tool path is a module constant and the capture path is created by this test."""
    _, out, _ = _run_tool("csvexport", timeout=300)
    return [row for row in csv.DictReader(io.StringIO(out)) if row["src_file"] == "kernel_profiler"]


# The workload's own subscriber counts every record the host decoded, which is the only place events and
# timestamped data are observable end to end: tracy-csvexport can export zones (-u/-g), messages (-m) and
# plots (-p), but not GPU markers, so the Tracy capture alone cannot prove the point-marker wire paths.
def _subscriber_totals(log: str) -> tuple[int, int, int]:
    m = re.search(r"subscriber saw (\d+) zones, (\d+) points, (\d+) stalls", log)
    assert m, f"workload printed no subscriber totals:\n{log[-2000:]}"
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# 10 named zones per iteration per RISC, plus the one firmware wrapper zone each RISC opens per launch.
def _expected_zones(gx: int, gy: int, iters: int) -> int:
    lanes = gx * gy * RISCS_PER_CORE
    return lanes * ZONES_PER_ITER * iters + lanes


# Only the data-movement kernels emit the point-marker trio (_Event, _Data, _Iter) under --markers 1.
def _expected_points(gx: int, gy: int, iters: int) -> int:
    return gx * gy * DM_RISCS_PER_CORE * iters * MARKERS_PER_ITER


def _run_workload(which: str, env_extra: dict) -> tuple[int, str]:
    env = dict(os.environ)
    env["TT_METAL_STREAMING_PROFILER"] = "1"  # its own mode; excludes TT_METAL_DEVICE_PROFILER
    env.update(env_extra)
    # --markers 1: nothing else in the tree emits PP_EVENT / Data payload records, so without it those wire
    # layouts go untested.
    rc, out, err = _run_tool(which, env=env, cwd=str(TT_METAL_HOME), timeout=300)
    return rc, out + err


def _skip_unless_streaming_started(log: str) -> None:
    # Match only the role-agnostic substring: a guard tied to fuller wording skips unconditionally the moment
    # the message is reworded, leaving the test green and asserting nothing.
    if "not Blackhole" in log or "[streaming profiler] active on" not in log:
        pytest.skip("streaming profiler did not start the DRISC relay (not Blackhole / no DRAM programmable cores)")


@pytest.mark.parametrize("gx,gy,iters", [(GRID_X, GRID_Y, SMOKE_ITERS)])
def test_streaming_profiler_workload(gx, gy, iters):
    """The pipeline still works: relays go resident and every record the device produced reaches the host.

    No Tracy here on purpose -- this is the regression guard for the streaming profiler itself, so a Tracy
    or capture-tool problem cannot mask a profiler one. Counts are exact because the wire is specified as
    lossless; a >= assertion would pass while silently dropping records.
    """
    assert WORKLOAD_BIN.exists(), f"workload not built: {WORKLOAD_BIN} (build target test_streaming_profiler_zones)"

    rc, log = _run_workload("workload-smoke", {})
    _skip_unless_streaming_started(log)
    assert rc == 0, f"workload failed (rc={rc}):\n{log[-2000:]}"

    zones, points, stalls = _subscriber_totals(log)
    assert stalls == 0, f"producers stalled ({stalls}); the relays are not keeping up:\n{log[-2000:]}"
    assert zones == _expected_zones(
        gx, gy, iters
    ), f"zone records lost: got {zones}, want {_expected_zones(gx, gy, iters)}"
    assert points == _expected_points(
        gx, gy, iters
    ), f"point records (events + timestamped data) lost: got {points}, want {_expected_points(gx, gy, iters)}"


@pytest.mark.parametrize("gx,gy,iters", [(GRID_X, GRID_Y, CAPTURE_ITERS)])
def test_streaming_profiler_zones_capture(gx, gy, iters):
    assert WORKLOAD_BIN.exists(), f"workload not built: {WORKLOAD_BIN} (build target test_streaming_profiler_zones)"
    assert TOOLS["capture"].is_file(), f"tracy-capture not found: {TOOLS['capture']} (build with ENABLE_TRACY=ON)"

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CAPTURE_FILE.unlink(missing_ok=True)
    port = _free_port()

    cap = subprocess.Popen(
        [_tool("capture"), "-o", str(CAPTURE_FILE), "-f", "-p", str(int(port))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # let tracy-capture start listening

    try:
        # Same run as the no-Tracy case, so it goes through the same runner: the sink is opt-in and the
        # capture is listening on TRACY_PORT.
        rc, log = _run_workload("workload-capture", {"TRACY_PORT": port, "TT_METAL_STREAMING_PROFILER_TRACY": "1"})
    finally:
        try:
            cap.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            cap.terminate()
            cap.communicate()

    _skip_unless_streaming_started(log)
    assert rc == 0, f"workload failed (rc={rc}):\n{log[-2000:]}"
    assert "active on 1 device(s)" in log, "streaming profiler did not report active"
    assert CAPTURE_FILE.exists() and CAPTURE_FILE.stat().st_size > 4096, "no/empty Tracy capture produced"

    if not TOOLS["csvexport"].is_file():
        pytest.fail(
            f"tracy-csvexport not built at {TOOLS['csvexport']} -- the device-zone assertion cannot run and this test "
            f"would otherwise verify only that the capture exceeds 4096 bytes."
        )

    rows = _device_zone_rows()
    lanes = len({row["thread"] for row in rows})
    assert lanes >= gx * gy * RISCS_PER_CORE, f"expected >= {gx * gy * RISCS_PER_CORE} lanes with zones, got {lanes}"

    # Names must come back resolved, not blank or hashed: the ids on the wire are structural (tu-id + local
    # id) and the host resolves them per ELF from .tt_zone_meta, so an empty or missing name means that
    # resolution broke even though zones still arrived.
    names = {row["name"] for row in rows}
    assert "" not in names, "a device zone came back unnamed -- ELF zone-name resolution failed"
    for tag in ("BR", "NC", "T0", "T1", "T2"):
        missing = {f"{tag}_Zone{i}" for i in range(ZONES_PER_ITER)} - names
        assert not missing, f"{tag}: device zones missing from the capture: {sorted(missing)}"

    # Events and timestamped data are GPU markers, which tracy-csvexport cannot export; the subscriber's
    # own totals are where the point-marker paths are observable.
    zones, points, stalls = _subscriber_totals(log)
    assert stalls == 0, f"producers stalled ({stalls})"
    assert zones == _expected_zones(
        gx, gy, iters
    ), f"zone records lost: got {zones}, want {_expected_zones(gx, gy, iters)}"
    assert points == _expected_points(
        gx, gy, iters
    ), f"point records (events + timestamped data) lost: got {points}, want {_expected_points(gx, gy, iters)}"
