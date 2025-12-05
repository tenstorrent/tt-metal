# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Consolidated test suite for the real-time profiler.

Each test in this file exercises a different axis of the real-time profiler
and can be selected individually with pytest ``-k``:

* ``test_callback``                  — Smoke test: register a Python callback,
                                       run a matmul, verify records arrive.
* ``test_no_short_zones``            — Trace capture / replay; verify no
                                       bogus <10µs "short" records appear in
                                       the callback stream (regression for the
                                       stale-end-timestamp bug).
* ``test_cross_reference``           — Run ResNet50 with both the real-time
                                       profiler and the device profiler, and
                                       cross-reference per-program kernel
                                       durations.
* ``test_cross_reference_tg``        — Same as ``test_cross_reference`` but on
                                       a (8,4) TG mesh; auto-skips when not on
                                       a TG/Galaxy box.
* ``test_host_device_correlation``   — Launch a workload under Tracy capture
                                       and verify 1:1 correspondence between
                                       host ``EnqueueProgram`` TracyMessages
                                       and device real-time records.
* ``test_sync_accuracy``             — Launch a workload under Tracy capture
                                       and verify every host/device
                                       ``SYNC_CHECK``/``FINISH_SYNC`` pair is
                                       aligned to within ±10µs on the Tracy
                                       timeline.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import socket
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Common paths / constants
# ---------------------------------------------------------------------------

TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
PROFILER_BIN_DIR = TT_METAL_HOME / "build" / "tools" / "profiler" / "bin"
CAPTURE_TOOL = PROFILER_BIN_DIR / "capture-release"
CSVEXPORT_TOOL = PROFILER_BIN_DIR / "csvexport-release"

# Root artifact dir; each test writes into a named sub-directory.
ARTIFACTS_ROOT = TT_METAL_HOME / "generated" / "profiler" / "realtime_profiler_tests"

# External workload scripts that are re-executed in a subprocess under Tracy
# capture for the correlation / sync tests.  They are kept as standalone
# scripts because Tracy capture must see a fresh process.
WORKLOAD_DIR = Path(__file__).parent
CORRELATION_WORKLOAD = WORKLOAD_DIR / "host_device_correlation_workload.py"
SYNC_WORKLOAD = WORKLOAD_DIR / "realtime_profiler_sync_workload.py"
CROSS_REFERENCE_WORKLOAD = WORKLOAD_DIR / "cross_reference_workload.py"
MATMUL_WORKLOAD = WORKLOAD_DIR / "matmul_workload.py"

# Every test in this module runs its device-touching work in a subprocess.
# The UMD ``CHIP_IN_USE_*_PCIe`` lock is held for the lifetime of the first
# process that touches the device, so if the pytest parent ever opened a
# device then every subsequent subprocess workload would block waiting for
# the lock.  Keeping the parent device-free lets all 6 tests run in a
# single pytest invocation.


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _free_port() -> str | None:
    """Find a free TCP port suitable for Tracy capture."""
    ip = socket.gethostbyname(socket.gethostname())
    for port in range(8086, 8500):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, port))
            s.close()
            return str(port)
        except (PermissionError, OSError):
            continue
    return None


def _run_under_tracy(
    workload_script: Path,
    out_tracy: Path,
    log_path: Path,
    extra_env: dict | None = None,
    timeout_s: int = 600,
):
    """
    Launch `capture-release` in the background, run `workload_script` under
    it in a subprocess with stdout streamed to ``log_path`` (avoids the OS
    pipe buffer filling up and deadlocking the child), and return
    (workload_returncode, workload_stdout_text).
    """
    assert CAPTURE_TOOL.exists(), f"Tracy capture tool not found: {CAPTURE_TOOL}"

    port = _free_port()
    assert port is not None, "No available TCP port for Tracy capture"

    capture_proc = subprocess.Popen(
        [str(CAPTURE_TOOL), "-o", str(out_tracy), "-f", "-p", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # let capture-release start listening

    env = dict(os.environ)
    env["TRACY_PORT"] = port
    env.pop("TT_METAL_DEVICE_PROFILER", None)  # must be off so TracyMessages flow
    if extra_env:
        env.update(extra_env)

    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                ["python3", str(workload_script)],
                env=env,
                cwd=str(TT_METAL_HOME),
                timeout=timeout_s,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
    finally:
        try:
            capture_proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            capture_proc.terminate()
            capture_proc.communicate()

    return proc.returncode, log_path.read_text()


def _export_messages(tracy_file: Path, out_csv: Path):
    with open(out_csv, "w") as f:
        subprocess.run(
            [str(CSVEXPORT_TOOL), "-m", "-s", ";", str(tracy_file)],
            stdout=f,
            stderr=subprocess.DEVNULL,
            check=True,
        )


def _export_zones_unwrapped(tracy_file: Path, out_csv: Path):
    with open(out_csv, "w") as f:
        subprocess.run(
            [str(CSVEXPORT_TOOL), "-u", "-s", ";", str(tracy_file)],
            stdout=f,
            stderr=subprocess.DEVNULL,
            check=True,
        )


def _save_artifacts(test_name: str, **files: Path):
    """Copy files into ARTIFACTS_ROOT/<test_name>/ for post-mortem analysis."""
    dest = ARTIFACTS_ROOT / test_name
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        if src is not None and src.exists():
            shutil.copy2(src, dest / name)
    return dest


# ---------------------------------------------------------------------------
# 1. / 2. Matmul-workload helpers (shared by callback + short-zones tests)
# ---------------------------------------------------------------------------

SHORT_ZONE_THRESHOLD_US = 10.0


def _run_matmul_workload(tmp_path: Path, mode: str, timeout_s: int = 180) -> list[dict]:
    """
    Launch ``matmul_workload.py`` in a fresh subprocess, read back the RT
    records it dumps to disk, and return them as a list of dicts.
    """
    assert MATMUL_WORKLOAD.exists(), f"Workload script not found: {MATMUL_WORKLOAD}"

    rt_path = tmp_path / "rt_records.json"
    workload_log = tmp_path / "workload_output.log"

    env = dict(os.environ)
    env["MODE"] = mode
    env["RT_RECORDS_PATH"] = str(rt_path)

    with open(workload_log, "w") as log_f:
        proc = subprocess.run(
            ["python3", str(MATMUL_WORKLOAD)],
            env=env,
            cwd=str(TT_METAL_HOME),
            timeout=timeout_s,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    stdout = workload_log.read_text()
    print(f"\n--- Workload output (tail) ---\n{stdout[-2000:]}\n--- End workload output ---")

    if proc.returncode == 2:
        pytest.skip("Workload reported no devices available")
    assert proc.returncode == 0, f"Workload failed (rc={proc.returncode})"
    assert rt_path.exists(), "Workload did not write RT records file"

    with open(rt_path) as f:
        return json.load(f)


@pytest.mark.timeout(300)
def test_callback(tmp_path):
    """Register a Python callback, run a small matmul workload, verify records arrive."""
    snapshot = _run_matmul_workload(tmp_path, mode="simple")

    print(f"\nCollected {len(snapshot)} real-time profiler records")
    for i, rec in enumerate(snapshot[:5]):
        delta = rec["end_timestamp"] - rec["start_timestamp"]
        print(f"  [{i}] program={rec['program_id']} chip={rec['chip_id']} ticks={delta}")
    if len(snapshot) > 5:
        print(f"  ... and {len(snapshot) - 5} more")

    assert len(snapshot) > 0, "Expected at least one real-time profiler record"
    for rec in snapshot:
        assert rec["end_timestamp"] >= rec["start_timestamp"], "end_timestamp < start_timestamp"
        assert rec["frequency_ghz"] > 0, "frequency_ghz must be positive"

    out_file = tmp_path / "rt_records.json"
    _save_artifacts("test_callback", records=out_file)


@pytest.mark.timeout(300)
def test_no_short_zones(tmp_path):
    """
    Capture a 50-matmul trace, replay it 10 times and confirm that no
    real-time profiler record reports a kernel duration below
    ``SHORT_ZONE_THRESHOLD_US``. These bogus sub-10µs records historically
    appeared when TRISC0 wrote a stale end timestamp or when non-GO dispatch
    commands were miscounted as programs.
    """
    snapshot = _run_matmul_workload(tmp_path, mode="trace")

    short = []
    valid = 0
    for idx, r in enumerate(snapshot):
        if r["program_id"] == 0 or r["frequency_ghz"] <= 0:
            continue
        valid += 1
        dur_us = (r["end_timestamp"] - r["start_timestamp"]) / r["frequency_ghz"] / 1000.0
        if dur_us < SHORT_ZONE_THRESHOLD_US:
            short.append((idx, r["program_id"], dur_us))

    print(
        f"\nRT records: total={len(snapshot)}, valid(pid!=0)={valid}, short(<{SHORT_ZONE_THRESHOLD_US}us)={len(short)}"
    )
    for idx, pid, dur in short[:20]:
        r = snapshot[idx]
        print(f"  SHORT [{idx}] pid={pid} dur_us={dur:.3f} start={r['start_timestamp']} end={r['end_timestamp']}")

    _save_artifacts("test_no_short_zones", records=tmp_path / "rt_records.json")

    assert valid > 0, "No valid real-time profiler records (pid != 0) collected"
    assert not short, (
        f"{len(short)}/{valid} real-time profiler records had duration "
        f"< {SHORT_ZONE_THRESHOLD_US}µs (first: {short[:5]})"
    )


# ---------------------------------------------------------------------------
# 3./4. Cross-reference real-time profiler vs device profiler
# ---------------------------------------------------------------------------

DEVICE_ID_NUM_BITS = 10
RELATIVE_TOLERANCE = 0.20
DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MIN_DEVICE_KERNEL_DURATION_NS = 10_000
CROSS_REF_PASS_RATE = 0.50  # at least 50% of matched pairs must be within tolerance


def _decode_runtime_id(encoded_runtime_id: int) -> int:
    return (encoded_runtime_id >> DEVICE_ID_NUM_BITS) & 0x1FFFFF


# Env vars that must be set BEFORE MetalContext / RuntimeOptions are
# constructed (RuntimeOptions caches TT_METAL_DEVICE_PROFILER on first
# creation).  The cross-reference tests set these in the subprocess env
# they hand to the workload script, which guarantees fresh caching per
# test invocation without polluting the parent pytest process.
DEVICE_PROFILER_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_SYNC": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}


def _run_cross_reference_workload(
    tmp_path: Path, mesh_shape: tuple[int, int], timeout_s: int, require_galaxy: bool = False
):
    """
    Execute the cross-reference workload in a fresh python subprocess with
    the device-profiler env vars in place.  All device-touching (cluster
    check, GetNumAvailableDevices, open_mesh_device) happens inside the
    subprocess so the parent pytest process never acquires the PCIe lock.
    Returns (rt_records, dev_flat, workload_log) on success; bails out via
    ``pytest.skip`` / ``pytest.fail`` otherwise.
    """
    rt_path = tmp_path / "rt_records.json"
    dev_path = tmp_path / "dev_perf.json"
    workload_log = tmp_path / "workload_output.log"

    env = dict(os.environ)
    env.update(DEVICE_PROFILER_ENV)
    env["MESH_SHAPE"] = f"{mesh_shape[0]},{mesh_shape[1]}"
    env["RT_RECORDS_PATH"] = str(rt_path)
    env["DEV_PERF_PATH"] = str(dev_path)
    if require_galaxy:
        env["REQUIRE_GALAXY"] = "1"

    # Stream stdout straight to a file so the OS pipe buffer can't fill up
    # and deadlock the child (device profiler emits megabytes of info logs).
    with open(workload_log, "w") as log_f:
        proc = subprocess.run(
            ["python3", str(CROSS_REFERENCE_WORKLOAD)],
            env=env,
            cwd=str(TT_METAL_HOME),
            timeout=timeout_s,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    stdout = workload_log.read_text()
    print(f"\n--- Workload output (tail) ---\n{stdout[-4000:]}\n--- End workload output ---")

    if proc.returncode == 2:
        pytest.skip("Workload reported insufficient devices for requested mesh")
    if proc.returncode == 3:
        pytest.skip("Workload reported non-Galaxy cluster (REQUIRE_GALAXY)")
    assert proc.returncode == 0, f"Cross-reference workload failed (rc={proc.returncode})"

    if not rt_path.exists() or not dev_path.exists():
        pytest.fail(f"Workload finished but did not write outputs: rt={rt_path.exists()} dev={dev_path.exists()}")

    with open(rt_path) as f:
        rt_records = json.load(f)
    with open(dev_path) as f:
        dev_flat = json.load(f)

    return rt_records, dev_flat, workload_log


def _cross_reference_impl(
    tmp_path: Path,
    mesh_shape: tuple[int, int],
    test_name: str,
    expected_multi_chip: bool,
    timeout_s: int = 900,
    require_galaxy: bool = False,
):
    """
    Shared body of the cross-reference tests.  Runs the workload in a
    subprocess, then compares RT durations to device profiler kernel
    durations (``DEVICE_KERNEL_DURATION_KEY``).
    """
    rt_snapshot, dev_flat, workload_log = _run_cross_reference_workload(
        tmp_path, mesh_shape, timeout_s, require_galaxy=require_galaxy
    )

    assert len(rt_snapshot) > 0, "No real-time profiler records collected"
    assert dev_flat, "No device profiler data returned"

    # Build duration maps.
    dev_by_raw = defaultdict(list)
    dev_by_decoded = defaultdict(list)
    for entry in dev_flat:
        duration_ns = entry["duration_ns"]
        if duration_ns <= 0:
            continue
        runtime_id = entry["runtime_id"]
        dev_by_raw[runtime_id].append(duration_ns)
        dev_by_decoded[_decode_runtime_id(runtime_id)].append(duration_ns)

    assert dev_by_raw or dev_by_decoded, "No device profiler programs with kernel duration data"

    rt_by_id = defaultdict(list)
    for rec in rt_snapshot:
        pid = rec["program_id"]
        freq = rec["frequency_ghz"]
        if pid == 0 or freq <= 0:
            continue
        dur_ns = (rec["end_timestamp"] - rec["start_timestamp"]) / freq
        if dur_ns <= 0:
            continue
        rt_by_id[pid].append(dur_ns)

    assert rt_by_id, "No valid real-time profiler records"

    # Choose better matching strategy.
    raw_matches = sum(1 for pid in rt_by_id if pid in dev_by_raw)
    decoded_matches = sum(1 for pid in rt_by_id if pid in dev_by_decoded)
    if raw_matches >= decoded_matches:
        dev_durations, match_strategy = dev_by_raw, "raw"
    else:
        dev_durations, match_strategy = dev_by_decoded, "decoded"

    # Cross-reference.
    matched = 0
    within = 0
    skipped_short = 0
    details = []
    for pid, rt_durs in sorted(rt_by_id.items()):
        if pid not in dev_durations:
            continue
        dev_durs = dev_durations[pid]
        for i in range(min(len(rt_durs), len(dev_durs))):
            rt_ns = rt_durs[i]
            dev_ns = dev_durs[i]
            rel_err = abs(rt_ns - dev_ns) / dev_ns if dev_ns > 0 else float("inf")
            short = dev_ns < MIN_DEVICE_KERNEL_DURATION_NS
            if short:
                skipped_short += 1
            ok = rel_err <= RELATIVE_TOLERANCE
            if not short:
                matched += 1
                if ok:
                    within += 1
            details.append(
                {
                    "program_id": pid,
                    "instance": i,
                    "rt_duration_ns": round(rt_ns, 1),
                    "dev_duration_ns": round(dev_ns, 1),
                    "relative_error": round(rel_err, 4),
                    "within_tolerance": ok,
                    "below_threshold": short,
                }
            )

    diagnostics = {
        "match_strategy": match_strategy,
        "rt_record_count": len(rt_snapshot),
        "rt_unique_program_ids": len(rt_by_id),
        "dev_unique_program_ids_raw": len(dev_by_raw),
        "dev_unique_program_ids_decoded": len(dev_by_decoded),
        "matched_pairs": matched,
        "skipped_short": skipped_short,
        "within_tolerance": within,
        "tolerance": RELATIVE_TOLERANCE,
        "min_device_kernel_duration_ns": MIN_DEVICE_KERNEL_DURATION_NS,
        "comparisons": details,
    }
    out_file = tmp_path / "cross_reference.json"
    with open(out_file, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n=== {test_name} ===")
    print(f"  Match strategy:          {match_strategy}")
    print(f"  RT records:              {len(rt_snapshot)}")
    print(f"  RT unique program IDs:   {len(rt_by_id)}")
    print(f"  Dev unique program IDs:  {len(dev_by_raw)} (raw) / {len(dev_by_decoded)} (decoded)")
    print(f"  Matched (>={MIN_DEVICE_KERNEL_DURATION_NS}ns): {matched} (skipped {skipped_short} short)")
    print(f"  Within {RELATIVE_TOLERANCE*100:.0f}% tolerance:  {within}/{matched}")
    print(f"  Diagnostics:             {out_file}")

    for d in details:
        tag = "skip" if d["below_threshold"] else ("OK" if d["within_tolerance"] else "FAIL")
        print(
            f"    pid={d['program_id']:>6} [{d['instance']}]: "
            f"rt={d['rt_duration_ns']:>12.1f} ns  dev={d['dev_duration_ns']:>12.1f} ns  "
            f"err={d['relative_error']*100:>6.2f}%  {tag}"
        )

    _save_artifacts(test_name, **{"cross_reference.json": out_file, "workload_output.log": workload_log})

    assert matched > 0, (
        f"No program IDs matched between real-time profiler and device profiler; "
        f"RT IDs sample: {sorted(rt_by_id)[:10]}, Dev raw: {sorted(dev_by_raw)[:10]}, "
        f"Dev decoded: {sorted(dev_by_decoded)[:10]}"
    )
    pass_rate = within / matched
    assert pass_rate >= CROSS_REF_PASS_RATE, (
        f"Only {within}/{matched} ({pass_rate*100:.1f}%) pairs within "
        f"{RELATIVE_TOLERANCE*100:.0f}% tolerance (need >= {CROSS_REF_PASS_RATE*100:.0f}%); see {out_file}"
    )

    if expected_multi_chip:
        chips_with_rt = {rec["chip_id"] for rec in rt_snapshot}
        assert (
            len(chips_with_rt) > 1
        ), f"Expected RT data from multiple chips on TG, got only {len(chips_with_rt)}: {sorted(chips_with_rt)}"


@pytest.mark.timeout(1200)
def test_cross_reference(tmp_path):
    """Cross-reference RT profiler vs device profiler on a single device."""
    if not CROSS_REFERENCE_WORKLOAD.exists():
        pytest.fail(f"Workload script not found: {CROSS_REFERENCE_WORKLOAD}")

    _cross_reference_impl(
        tmp_path,
        mesh_shape=(1, 1),
        test_name="test_cross_reference",
        expected_multi_chip=False,
        timeout_s=900,
    )


TG_MESH_SHAPE = (8, 4)


@pytest.mark.timeout(2100)
def test_cross_reference_tg(tmp_path):
    """Cross-reference RT profiler vs device profiler across a full TG mesh."""
    if not CROSS_REFERENCE_WORKLOAD.exists():
        pytest.fail(f"Workload script not found: {CROSS_REFERENCE_WORKLOAD}")

    _cross_reference_impl(
        tmp_path,
        mesh_shape=TG_MESH_SHAPE,
        test_name="test_cross_reference_tg",
        expected_multi_chip=True,
        timeout_s=1800,
        require_galaxy=True,
    )


# ---------------------------------------------------------------------------
# 5. Host/device 1:1 correlation (runs workload under Tracy capture)
# ---------------------------------------------------------------------------


def _parse_enqueue_op_ids(csv_path: Path) -> list[int]:
    """Extract EnqueueProgram op_id=<N> messages from csvexport -m output."""
    op_ids = []
    pattern = re.compile(r"EnqueueProgram op_id=(\d+)")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar="`")
        for row in reader:
            m = pattern.match(row.get("MessageName", ""))
            if m:
                op_ids.append(int(m.group(1)))
    return op_ids


@pytest.mark.timeout(900)
def test_host_device_correlation(tmp_path):
    """
    Run the correlation workload under Tracy capture. Verify every host
    ``EnqueueProgram op_id=X`` Tracy message has a matching device-side
    real-time profiler record for ``program_id=X`` with identical
    multiplicity.
    """
    assert CAPTURE_TOOL.exists(), f"Tracy capture tool not found: {CAPTURE_TOOL}"
    assert CSVEXPORT_TOOL.exists(), f"Tracy csvexport tool not found: {CSVEXPORT_TOOL}"
    assert CORRELATION_WORKLOAD.exists(), f"Workload script not found: {CORRELATION_WORKLOAD}"

    tracy_file = tmp_path / "trace.tracy"
    messages_csv = tmp_path / "messages.csv"
    device_records_json = tmp_path / "device_records.json"
    workload_log = tmp_path / "workload_output.log"

    rc, stdout = _run_under_tracy(
        CORRELATION_WORKLOAD,
        tracy_file,
        log_path=workload_log,
        extra_env={"DEVICE_RECORDS_PATH": str(device_records_json)},
        timeout_s=600,
    )
    print(f"\n--- Workload output (tail) ---\n{stdout[-4000:]}\n--- End workload output ---")
    assert rc == 0, f"Workload failed (rc={rc})"
    assert tracy_file.exists(), f"Tracy trace file not generated: {tracy_file}"
    assert device_records_json.exists(), f"Device records file not written: {device_records_json}"

    _export_messages(tracy_file, messages_csv)
    assert messages_csv.stat().st_size > 0, "Messages CSV is empty"

    host_op_ids = _parse_enqueue_op_ids(messages_csv)
    with open(device_records_json, "r") as f:
        device_records = json.load(f)

    assert len(host_op_ids) > 0, "No EnqueueProgram TracyMessages found"
    assert len(device_records) > 0, "No device records collected"

    device_program_ids = [r["program_id"] for r in device_records]
    host_set = set(host_op_ids)
    dev_set = set(device_program_ids)
    host_counts = Counter(host_op_ids)
    dev_counts = Counter(device_program_ids)

    print(f"\n  Host op_ids: {len(host_op_ids)} total, range {min(host_op_ids)}-{max(host_op_ids)}")
    print(f"  Device PIDs: {len(device_program_ids)} total, {len(dev_set)} unique")
    print(f"  First 15 device PIDs: {device_program_ids[:15]}")
    print(f"  Last 15 device PIDs:  {device_program_ids[-15:]}")

    missing_on_device = sorted(host_set - dev_set)
    # The very last dispatched program may be missing because dispatch_s's
    # TERMINATE iteration consumes the final FIFO entry before the profiler
    # core can push the record. Allow exactly one missing ID if it's the last.
    if len(missing_on_device) == 1 and missing_on_device[0] == max(host_op_ids):
        print(f"  NOTE: last program_id {missing_on_device[0]} missing (TERMINATE edge case)")
    else:
        assert not missing_on_device, (
            f"{len(missing_on_device)} host op_id(s) not found in device records: "
            f"{missing_on_device[:10]}{'...' if len(missing_on_device) > 10 else ''}"
        )

    infra_only = sorted(dev_set - host_set)
    if infra_only:
        print(f"  INFO: {len(infra_only)} device-only program_id(s) (infrastructure): {infra_only}")

    matched_ids = host_set & dev_set
    mismatched = [
        (pid, host_counts[pid], dev_counts[pid]) for pid in sorted(matched_ids) if host_counts[pid] != dev_counts[pid]
    ]
    assert not mismatched, (
        f"Multiplicity mismatch for {len(mismatched)} program_id(s): "
        f"{[(pid, f'host={h}, device={d}') for (pid, h, d) in mismatched[:10]]}"
    )

    matched_host_count = sum(host_counts[pid] for pid in matched_ids)
    allowed_missing = 1 if len(missing_on_device) == 1 and missing_on_device[0] == max(host_op_ids) else 0
    assert (
        matched_host_count >= len(host_op_ids) - allowed_missing
    ), f"Not all host messages matched: {matched_host_count}/{len(host_op_ids)}"

    # Sanity: kernel record timestamps.  Allow a small negative delta for the
    # deterministic startup race where the compute kernel detects dispatch_d's
    # stream-register clearing before dispatch_s records the first start.
    startup_race_threshold = 100_000
    for rec in device_records:
        if rec["program_id"] in matched_ids:
            delta = int(rec["end_timestamp"]) - int(rec["start_timestamp"])
            assert delta >= -startup_race_threshold, (
                f"Invalid timestamps for program_id={rec['program_id']}: "
                f"end={rec['end_timestamp']} < start={rec['start_timestamp']} (delta={delta})"
            )
        assert rec["frequency_ghz"] > 0, f"Invalid frequency for program_id={rec['program_id']}"

    print(f"\n  Matched user programs: {len(matched_ids)}")
    print(f"  Host TracyMessages:    {len(host_op_ids)}")
    print(f"  Device records:        {len(device_records)}")

    _save_artifacts(
        "test_host_device_correlation",
        **{
            "trace.tracy": tracy_file,
            "messages.csv": messages_csv,
            "device_records.json": device_records_json,
            "workload_output.log": workload_log,
        },
    )


# ---------------------------------------------------------------------------
# 6. Host/device sync accuracy (runs workload under Tracy capture)
# ---------------------------------------------------------------------------

SYNC_DIFF_THRESHOLD_NS = 10_000  # ±10 µs
SYNC_PAIRING_WINDOW_NS = 100_000  # ±100 µs — gross-error safety window

HOST_SYNC_MESSAGES = {"SYNC_CHECK", "FINISH_SYNC"}


def _parse_host_sync_messages(csv_path: Path) -> list[tuple[str, int]]:
    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            name, ns = row[0], row[1]
            if name in HOST_SYNC_MESSAGES:
                out.append((name, int(ns)))
    return out


def _parse_device_sync_zones(csv_path: Path) -> list[int]:
    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row.get("name") == "SYNC_CHECK" and row.get("src_file") == "sync_check":
                try:
                    out.append(int(row["ns_since_start"]))
                except (KeyError, ValueError):
                    continue
    out.sort()
    return out


def _pair_sync(host_msgs, device_zones):
    """For each host sync message, pick the closest device SYNC_CHECK zone."""
    pairs = []
    for name, h_ns in host_msgs:
        if not device_zones:
            pairs.append((name, h_ns, None, None))
            continue
        # Binary search for insertion point.
        lo, hi = 0, len(device_zones) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if device_zones[mid] < h_ns:
                lo = mid + 1
            else:
                hi = mid
        candidates = []
        if lo < len(device_zones):
            candidates.append(device_zones[lo])
        if lo > 0:
            candidates.append(device_zones[lo - 1])
        best = min(candidates, key=lambda d: abs(d - h_ns))
        pairs.append((name, h_ns, best, best - h_ns))
    return pairs


@pytest.mark.timeout(600)
def test_sync_accuracy(tmp_path):
    """
    Run a short workload under Tracy capture, pair each host
    ``SYNC_CHECK``/``FINISH_SYNC`` Tracy message with the nearest
    device-side ``SYNC_CHECK`` GPU zone on the timeline, and verify each
    pair is within ±10µs (i.e. H2D PCIe write latency).
    """
    assert CAPTURE_TOOL.exists(), f"Tracy capture tool not found: {CAPTURE_TOOL}"
    assert CSVEXPORT_TOOL.exists(), f"Tracy csvexport tool not found: {CSVEXPORT_TOOL}"
    assert SYNC_WORKLOAD.exists(), f"Workload script not found: {SYNC_WORKLOAD}"

    tracy_file = tmp_path / "trace.tracy"
    messages_csv = tmp_path / "messages.csv"
    zones_csv = tmp_path / "zones.csv"
    workload_log = tmp_path / "workload_output.log"

    rc, stdout = _run_under_tracy(SYNC_WORKLOAD, tracy_file, log_path=workload_log, timeout_s=300)
    print(f"\n--- Workload output (tail) ---\n{stdout[-4000:]}\n--- End workload output ---")
    assert rc == 0, f"Workload failed (rc={rc})"
    assert tracy_file.exists(), f"Tracy trace file not generated: {tracy_file}"

    _export_messages(tracy_file, messages_csv)
    _export_zones_unwrapped(tracy_file, zones_csv)

    host_msgs = _parse_host_sync_messages(messages_csv)
    device_zones = _parse_device_sync_zones(zones_csv)
    assert host_msgs, "No host SYNC_CHECK/FINISH_SYNC messages in trace"
    assert device_zones, (
        "No device SYNC_CHECK GPU zones found. If csvexport was built without the "
        "GPU-zone fix the zones won't be emitted — rebuild csvexport-release."
    )

    pairs = _pair_sync(host_msgs, device_zones)
    print(f"\nMatched {len(pairs)} host sync message(s) against {len(device_zones)} device SYNC_CHECK zone(s).")
    for name, h_ns, d_ns, diff in pairs:
        if d_ns is None:
            print(f"  {name:>12} @ {h_ns:>14,}ns  -> NO MATCH")
        else:
            print(f"  {name:>12} @ {h_ns:>14,}ns  -> {d_ns:>14,}ns  diff={diff:+.0f}ns")

    _save_artifacts(
        "test_sync_accuracy",
        **{
            "trace.tracy": tracy_file,
            "messages.csv": messages_csv,
            "zones.csv": zones_csv,
            "workload_output.log": workload_log,
        },
    )

    missing = [(n, h) for (n, h, d, _diff) in pairs if d is None]
    assert not missing, f"{len(missing)} host messages had no matching device zone: {missing[:5]}"

    far = [(n, h, d, diff) for (n, h, d, diff) in pairs if abs(diff) > SYNC_PAIRING_WINDOW_NS]
    assert not far, (
        f"{len(far)} pair(s) outside ±{SYNC_PAIRING_WINDOW_NS}ns window "
        f"(dropped message or gross mis-calibration): {far[:5]}"
    )

    bad = [(n, h, d, diff) for (n, h, d, diff) in pairs if abs(diff) > SYNC_DIFF_THRESHOLD_NS]
    assert not bad, f"{len(bad)}/{len(pairs)} sync pair(s) exceeded ±{SYNC_DIFF_THRESHOLD_NS}ns:\n" + "\n".join(
        f"  {n} host={h}ns device={d}ns diff={diff:+.0f}ns" for (n, h, d, diff) in bad
    )

    diffs = [diff for (_n, _h, _d, diff) in pairs]
    print(
        f"\nAll {len(pairs)} sync pairs within ±{SYNC_DIFF_THRESHOLD_NS}ns "
        f"(min={min(diffs):+.0f}ns, max={max(diffs):+.0f}ns, mean={sum(diffs)/len(diffs):+.0f}ns)."
    )
