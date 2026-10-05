#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from tracy.perf_counter_multipass import (
    PERF_COUNTER_L1_GROUPS,
    PERF_COUNTER_MAX_GROUPS_PER_PASS,
    arch_l1_groups,
    merge_perf_counter_device_logs,
    perf_counter_groups_to_bitfield,
    schedule_perf_counter_passes,
    run_perf_counter_passes,
)

CSV_HEADER = (
    "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, "
    "trace id, trace id counter, zone name, type, source line, source file, meta data\n"
)


def row(*fields):
    """A device log row: the eight leading fields, then the trace columns and the rest."""
    return ",".join(str(f) for f in fields) + ",,,,,,,\n"


def assert_pass_invariants(passes):
    for p in passes:
        assert len(p) <= PERF_COUNTER_MAX_GROUPS_PER_PASS
        assert sum(g in PERF_COUNTER_L1_GROUPS for g in p) <= 1


def test_full_blackhole_set_schedules_one_pass_per_l1_bank():
    groups = ["fpu", "pack", "unpack", "instrn", "l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"]
    passes = schedule_perf_counter_passes(groups)
    assert len(passes) == 6
    assert_pass_invariants(passes)
    assert sorted(g for p in passes for g in p) == sorted(groups)


def test_wormhole_all_set_schedules_two_passes():
    groups = ["fpu", "pack", "unpack", "instrn", "l1_0", "l1_1"]
    passes = schedule_perf_counter_passes(groups)
    assert len(passes) == 2
    assert_pass_invariants(passes)
    assert sorted(g for p in passes for g in p) == sorted(groups)


def test_requests_that_fit_stay_single_pass():
    assert schedule_perf_counter_passes(["fpu", "pack", "unpack"]) == [["fpu", "pack", "unpack"]]
    assert len(schedule_perf_counter_passes(["fpu", "instrn", "l1_0"])) == 1


def test_two_l1_banks_force_two_passes():
    assert len(schedule_perf_counter_passes(["l1_0", "l1_1"])) == 2


def test_group_cap_forces_extra_pass():
    assert len(schedule_perf_counter_passes(["fpu", "pack", "unpack", "instrn"])) == 2


def test_dedup_case_insensitive_and_empty():
    assert schedule_perf_counter_passes(["FPU", "fpu", "Pack"]) == [["fpu", "pack"]]
    assert schedule_perf_counter_passes([]) == []


def test_bitfield_matches_perf_counters_hpp_bits():
    assert perf_counter_groups_to_bitfield(["fpu", "pack", "unpack", "l1_0", "instrn"]) == 47
    assert perf_counter_groups_to_bitfield(["l1_4"]) == 1 << 8
    assert perf_counter_groups_to_bitfield(["l1_5"]) == 1 << 9


def test_arch_l1_groups():
    assert arch_l1_groups(True) == ["l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"]
    assert arch_l1_groups(False, is_quasar=True) == []
    assert arch_l1_groups(False) == ["l1_0", "l1_1"]


FPU = ["FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"]
PACK = [
    "PACKER0_DEST_READ_REQ",
    "PACKER_BUSY",
    "DEST_READ_GRANTED_0",
    "MATH_NOT_STALLED_DEST_WR_PORT",
    "MATH_NOT_SCOREBOARD_STALLED",
]
IDENTITY = [0, 1, 1, "BRISC", 7, None, None]


def counter_row(identity, counter, timestamp=110):
    chip, x, y, risc, op, trace, replay = identity
    metadata = json.dumps({"counter type": counter, "ref cnt": 100, "value": 42}).replace(",", ";")
    fields = [
        chip,
        x,
        y,
        risc,
        9090,
        timestamp,
        42,
        op,
        "" if trace is None else trace,
        "" if replay is None else replay,
        "",
        "TS_DATA",
        0,
        "",
        metadata,
    ]
    return ",".join(map(str, fields)) + "\n"


def make_capture(tmp_path, identities=None):
    identities = identities or [IDENTITY]
    paths, manifests = [], []
    for i, counters in enumerate((FPU, PACK)):
        path = tmp_path / f"pass_{i}.csv"
        path.write_text(
            "ARCH: blackhole\n"
            + CSV_HEADER
            + "".join(
                counter_row(key, counter, 110 + 100 * i + 1000 * n)
                for n, key in enumerate(identities)
                for counter in counters
            )
        )
        paths.append(path)
        manifests.append(
            {
                "schema_version": 1,
                "arch": "blackhole",
                "bitfield": [1, 2][i],
                "device_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executed_readouts": identities,
            }
        )
    return paths, manifests


def rebind(path, manifest):
    # Synthetic independent execution evidence stays fixed when mutating CSV coverage.
    manifest["device_log_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_disjoint_groups_and_replay_anchors(tmp_path):
    ids = [IDENTITY, [0, 1, 1, "BRISC", 8, 3, 0], [0, 1, 1, "BRISC", 8, 3, 1], [1, 2, 2, "BRISC", 9, None, None]]
    paths, evidence = make_capture(tmp_path, ids)
    original = [p.read_bytes() for p in paths]
    out = tmp_path / "merged.csv"
    merge_perf_counter_device_logs(paths, out, [1, 2], evidence)
    lines = out.read_text().splitlines(keepends=True)
    assert "".join(lines[:14]) == original[0].decode()
    assert lines[14:] == [counter_row(key, c, 110 + 1000 * n) for n, key in enumerate(ids) for c in PACK]
    assert [p.read_bytes() for p in paths] == original


@pytest.mark.parametrize(
    "fault",
    [
        "extra_operation",
        "missing_operation",
        "missing_chip",
        "missing_replay",
        "no_counter_rows",
        "missing_core",
        "wrong_risc",
        "missing_counter",
        "duplicate",
    ],
)
def test_incomplete_capture_rejected_without_publishing(tmp_path, fault):
    ids = [
        IDENTITY,
        [0, 1, 1, "BRISC", 8, None, None],
        [1, 1, 1, "BRISC", 7, None, None],
        [0, 1, 1, "BRISC", 7, 3, 1],
        [0, 2, 1, "BRISC", 7, None, None],
    ]
    paths, evidence = make_capture(tmp_path, ids)
    lines = paths[1].read_text().splitlines(keepends=True)
    if fault == "extra_operation":
        lines.append(counter_row([0, 1, 1, "BRISC", 99, None, None], PACK[0]))
    elif fault in ("missing_operation", "missing_chip", "missing_replay", "missing_core"):
        index = {"missing_operation": 1, "missing_chip": 2, "missing_replay": 3, "missing_core": 4}[fault]
        del lines[2 + index * 5 : 2 + (index + 1) * 5]
    elif fault == "no_counter_rows":
        lines = lines[:2]
    elif fault == "wrong_risc":
        lines[2] = lines[2].replace("BRISC", "TRISC_0")
    elif fault == "missing_counter":
        del lines[2]
    else:
        lines.append(lines[2])
    paths[1].write_text("".join(lines))
    rebind(paths[1], evidence[1])
    out = tmp_path / "merged.csv"
    out.write_text("previous result\n")
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only contracts
        merge_perf_counter_device_logs(paths, out, [1, 2], evidence)
    assert out.read_text() == "previous result\n"


@pytest.mark.parametrize(
    "case", ["extra_operation", "missing_operation", "missing_chip", "missing_replay", "no_counter_rows"]
)
def test_run43_original_fixtures_fail_closed(tmp_path, case):
    # Same header and rows as run43/cpu_review.py; no independent execution evidence exists.
    def record(op=1, chip=0, replay=0, timer=9090):
        return f"{chip},0,0,BRISC,{timer},100,42,{op},0,{replay},counter,TS_DATA,1,kernel.cpp\n"

    base, later = record(), record()
    if case == "extra_operation":
        later += record(op=2)
    if case == "missing_operation":
        base += record(op=2)
    if case == "missing_chip":
        base += record(chip=1)
    if case == "missing_replay":
        base, later = record(replay=1) + record(replay=2), record(replay=1)
    if case == "no_counter_rows":
        later = record(timer=123)
    header = "ARCH: blackhole\nDEVICE ID,CORE X,CORE Y,RISC,TIMER ID,TIME[cycles since reset],STAT VALUE,RUN HOST ID,TRACE ID,TRACE ID COUNTER,ZONE NAME,ZONE PHASE,SOURCE LINE,SOURCE FILE\n"
    logs = tmp_path / ".logs"
    logs.mkdir()
    calls = []

    def workload(env):
        (logs / "profile_log_device.csv").write_text(header + (base if not calls else later))
        calls.append(env)

    assert run_perf_counter_passes(workload, {}, [1, 32], tmp_path) is False
    assert not (logs / "profile_log_device.csv").exists()
    assert (logs / "perf_counter_passes/pass_0.csv").read_text() == header + base


@pytest.mark.parametrize(
    "fault",
    ["absent", "stale_hash", "mask", "duplicate_identity", "different_execution", "empty", "arch", "malformed_counter"],
)
def test_unverifiable_evidence_fails_closed(tmp_path, fault):
    paths, evidence = make_capture(tmp_path)
    if fault == "absent":
        evidence = None
    elif fault == "stale_hash":
        evidence[1]["device_log_sha256"] = "0" * 64
    elif fault == "mask":
        evidence[1]["bitfield"] = 1
    elif fault == "duplicate_identity":
        evidence[1]["executed_readouts"] = [IDENTITY, IDENTITY]
    elif fault == "different_execution":
        evidence[1]["executed_readouts"] = [[0, 1, 1, "BRISC", 99, None, None]]
    elif fault == "empty":
        evidence[1]["executed_readouts"] = []
    elif fault == "arch":
        evidence[1]["arch"] = "quasar"
    else:
        paths[1].write_text(paths[1].read_text().replace('"counter type"', '"unknown"'))
        rebind(paths[1], evidence[1])
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only contracts
        merge_perf_counter_device_logs(paths, tmp_path / "merged.csv", [1, 2], evidence)


def test_runner_contract_preserves_raw_and_reports_success(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    paths, evidence = make_capture(source)
    logs = tmp_path / ".logs"
    logs.mkdir()
    calls = []

    def workload(env):
        i = len(calls)
        calls.append(env)
        (logs / "profile_log_device.csv").write_bytes(paths[i].read_bytes())
        Path(env["TT_METAL_PROFILER_EXECUTION_MANIFEST"]).write_text(json.dumps(evidence[i]))

    assert run_perf_counter_passes(workload, {}, [1, 2], tmp_path) is True
    result = json.loads((logs / "perf_counter_passes/merge_result.json").read_text())
    assert result["complete"] is True
    for i in range(2):
        assert (logs / f"perf_counter_passes/pass_{i}.csv").read_bytes() == paths[i].read_bytes()
    # Never overwrite raw evidence from a previous invocation.
    assert run_perf_counter_passes(workload, {}, [1, 2], tmp_path) is False
    assert len(calls) == 2


@pytest.mark.parametrize(
    "fault", ["correlated_missing_core", "wrong_counter", "unrequested_counter", "truncated", "raw_output"]
)
def test_common_loss_and_invalid_records_fail_closed(tmp_path, fault):
    identities = [IDENTITY, [1, 1, 1, "BRISC", 7, None, None]]
    paths, evidence = make_capture(tmp_path, identities)
    output = tmp_path / "merged.csv"
    if fault == "correlated_missing_core":
        for path, manifest in zip(paths, evidence):
            path.write_text(
                "\n".join(line for line in path.read_text().splitlines() if not line.startswith("1,")) + "\n"
            )
            rebind(path, manifest)
    elif fault == "wrong_counter":
        paths[0].write_text(paths[0].read_text().replace("FPU_COUNTER", "FPU_COUNTER_BAD"))
        rebind(paths[0], evidence[0])
    elif fault == "unrequested_counter":
        paths[0].write_text(paths[0].read_text() + counter_row(IDENTITY, PACK[0]))
        rebind(paths[0], evidence[0])
    elif fault == "truncated":
        paths[0].write_text(paths[0].read_text().rstrip())
        rebind(paths[0], evidence[0])
    else:
        output = paths[0]
    before = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only contracts
        merge_perf_counter_device_logs(paths, output, [1, 2], evidence)
    assert [p.read_bytes() for p in paths] == before


@pytest.mark.parametrize("failure", ["missing_log", "exception", "exit", "bad_manifest"])
def test_runner_failures_preserve_raw_logs_and_result(tmp_path, failure):
    logs = tmp_path / ".logs"
    logs.mkdir()

    def workload(env):
        if failure != "missing_log":
            (logs / "profile_log_device.csv").write_text("raw failed capture\n")
        if failure == "exception":
            raise RuntimeError("workload failed")
        if failure == "exit":
            raise SystemExit(4)
        if failure == "bad_manifest":
            Path(env["TT_METAL_PROFILER_EXECUTION_MANIFEST"]).write_text("{")

    assert run_perf_counter_passes(workload, {}, [1, 2], tmp_path) is False
    assert not (logs / "profile_log_device.csv").exists()
    result = json.loads((logs / "perf_counter_passes/merge_result.json").read_text())
    assert result["complete"] is False and result["error"]
    if failure != "missing_log":
        assert (logs / "perf_counter_passes/pass_0.csv").read_text() == "raw failed capture\n"


def test_cli_exits_four_on_failed_merge():
    # Execute the actual CLI dispatch block without importing native Tracy or launching processes.
    import ast
    import sys

    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "tools/tracy/__main__.py").read_text())
    block = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "pass_bitfields and len(pass_bitfields) > 1"
    )
    with pytest.raises(SystemExit) as failure:  # allow-pytest.raises: CPU-only contracts
        exec(
            compile(ast.Module(body=[block], type_ignores=[]), "cli-dispatch", "exec"),
            {
                "pass_bitfields": [1, 2],
                "run_perf_counter_passes": lambda *a: False,
                "run_workload": None,
                "envVars": {},
                "outputFolder": None,
                "sys": sys,
            },
        )
    assert failure.value.code == 4


@pytest.mark.parametrize("arch", ["blackhole", "wormhole_b0"])
def test_native_semantics_cover_every_scheduled_group(arch):
    from tracy.perf_counter_multipass import _expected_counter_types, resolve_perf_counter_groups
    from tracy.perf_counter_sizing import COUNTERS_PER_GROUP

    groups = resolve_perf_counter_groups(["all"], arch)
    for scheduled in schedule_perf_counter_passes(groups):
        assert len(_expected_counter_types(arch, scheduled)) == sum(COUNTERS_PER_GROUP[arch][g] for g in scheduled)
    assert _expected_counter_types(arch, ["fpu"]) == set(FPU)
    l1 = _expected_counter_types(arch, ["l1_0"])
    assert ("L1_0_UNPACKER_1_ECC" in l1) == (arch == "blackhole")
    assert ("L1_0_UNPACKER_1_ECC_PACK1" in l1) == (arch == "wormhole_b0")
    assert "L1_0_UNIFIED_PACKER" not in l1  # retired enum slot is not emitted
