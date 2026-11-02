# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for perf_headroom.py — wave breakdown + perf CSV -> headroom verdict."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parent / "perf_headroom.py"
_SPEC = importlib.util.spec_from_file_location("perf_headroom", _PATH)
perf_headroom = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_headroom)


def _rates(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


BASE_RATES = {
    "default": {"cycles": 1, "source": "single-issue per thread"},
    "opcodes": {"SFPMAD": {"cycles": 1, "source": "assembly.yaml SFPMAD"}},
}


def _thread(issue, stall=None, idle=0, by_opcode=None):
    return {
        "issue": issue,
        "stall": stall or {},
        "idle": idle,
        "unknown": 0,
        "instructions_issued": issue,
        "instructions_by_opcode": by_opcode or {},
    }


def _report(cycles, threads, status="complete"):
    return {
        "status": status,
        "window": {
            "begin_fs": 0,
            "end_fs": cycles * 10,
            "cycles": cycles,
            "source": "wallclock",
        },
        "clock_period_fs": 10,
        "threads": threads,
        "missing_signals": [],
        "unknown_time_fs": 0,
    }


def _csv_row(per_tile, **extra):
    row = {
        "mathop": "Sigmoid",
        "marker": "TILE_LOOP",
        "loop_factor": "32",
        "tile_cnt": "4",
        "mean(MATH_ISOLATE)": str(per_tile),
        "mean(L1_TO_L1)": str(per_tile + 50),
    }
    row.update(extra)
    return row


# --- rates -------------------------------------------------------------------


def test_load_rates_rejects_unsourced(tmp_path):
    bad = _rates(
        tmp_path,
        "r.json",
        {
            "default": {"cycles": 1, "source": "x"},
            "opcodes": {"SFPSWAP": {"cycles": 2, "source": ""}},
        },
    )
    with pytest.raises(ValueError, match="SFPSWAP"):
        perf_headroom.load_rates([bad])


def test_load_rates_rejects_unsourced_default(tmp_path):
    bad = _rates(tmp_path, "r.json", {"default": {"cycles": 1}, "opcodes": {}})
    with pytest.raises(ValueError, match="default"):
        perf_headroom.load_rates([bad])


def test_later_rates_override_earlier(tmp_path):
    first = _rates(tmp_path, "a.json", BASE_RATES)
    second = _rates(
        tmp_path,
        "b.json",
        {"opcodes": {"SFPMAD": {"cycles": 2, "source": "measured on wave"}}},
    )
    rates = perf_headroom.load_rates([first, second])
    assert rates["opcodes"]["SFPMAD"] == {"cycles": 2, "source": "measured on wave"}
    assert rates["default"]["cycles"] == 1


# --- floor -------------------------------------------------------------------


def test_unit_floor_uses_default_and_reports_it(tmp_path):
    rates = perf_headroom.load_rates([_rates(tmp_path, "r.json", BASE_RATES)])
    thread = _thread(6, by_opcode={"SFPMAD": 4, "FOO": 2})
    assert perf_headroom.unit_floor_cycles(thread, rates) == (6, ["FOO"])


def test_unit_floor_without_opcodes_defaults_everything(tmp_path):
    rates = perf_headroom.load_rates([_rates(tmp_path, "r.json", BASE_RATES)])
    assert perf_headroom.unit_floor_cycles(_thread(7), rates) == (7, ["<all>"])


def test_opcode_hex_keys_map_to_mnemonics(tmp_path):
    yaml_text = "SFPMAD:\n    op_binary: 0x84\n    ex_resource: SFPU\nSFPNOP:\n    op_binary: 0x8f\n"
    path = tmp_path / "assembly.yaml"
    path.write_text(yaml_text)
    table = perf_headroom.load_opcode_names(path)
    assert table == {"0x84": "SFPMAD", "0x8f": "SFPNOP"}
    assert perf_headroom.name_opcodes({"0x84": 3, "0x99": 1}, table) == {
        "SFPMAD": 3,
        "0x99": 1,
    }


def test_per_tile_matches_post_csv_formula():
    assert perf_headroom.per_tile(6400, 32, 4) == 50.0


# --- verdict -----------------------------------------------------------------


def _verdict(tmp_path, report, row, **kw):
    rates = perf_headroom.load_rates([_rates(tmp_path, "r.json", BASE_RATES)])
    return perf_headroom.headroom_verdict(
        report, row, rates, run_type="MATH_ISOLATE", **kw
    )


def test_verdict_reconciles_and_finds_bottleneck(tmp_path):
    # 128 tile-iterations; math issues 20/tile, stalls 25/tile, idle 5/tile -> 50/tile
    report = _report(
        6400,
        {
            "math": _thread(2560, stall={"srcb_dvalid": 3200}, idle=640),
            "sfpu": _thread(1280, stall={"semaphore": 640}, idle=4480),
        },
    )
    verdict = _verdict(tmp_path, report, _csv_row(50.0))
    assert verdict["reconcile"] == {
        "wave_per_tile": 50.0,
        "csv_per_tile": 50.0,
        "delta_pct": 0.0,
        "ok": True,
    }
    assert verdict["bottleneck"] == "math"
    assert verdict["units"]["math"]["floor_per_tile"] == 20.0
    assert verdict["stall_headroom_pct"] == 60.0
    assert verdict["causes"][0] == ["math.srcb_dvalid", 25.0]
    values = [c[1] for c in verdict["causes"]]
    assert values == sorted(values, reverse=True)
    assert verdict["instruction_headroom"] is None


def test_instruction_headroom(tmp_path):
    report = _report(6400, {"sfpu": _thread(2560, idle=3840)})
    verdict = _verdict(
        tmp_path, report, _csv_row(50.0), best_count=15, best_count_source="hand count"
    )
    assert verdict["instruction_headroom"] == {
        "current": 20.0,
        "best": 15,
        "source": "hand count",
        "pct": 25.0,
    }


def test_reconcile_fails_over_two_percent(tmp_path):
    report = _report(6400 * 103 // 100, {"math": _thread(100)})
    verdict = _verdict(tmp_path, report, _csv_row(50.0))
    assert verdict["reconcile"]["ok"] is False


def test_incomplete_report_is_refused(tmp_path):
    report = _report(6400, {}, status="no_window")
    with pytest.raises(ValueError, match="no_window"):
        _verdict(tmp_path, report, _csv_row(50.0))


# --- CLI ---------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> Path:
    header = list(rows[0])
    lines = [",".join(header)] + [",".join(r[h] for h in header) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _cli(tmp_path, csv_rows, select, cycles=6400):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(cycles, {"math": _thread(2560, idle=3840)})))
    csv_path = _write_csv(tmp_path / "x.post.csv", csv_rows)
    rates = _rates(tmp_path, "r.json", BASE_RATES)
    argv = [
        sys.executable,
        str(_PATH),
        "--wave-report",
        str(report),
        "--perf-csv",
        str(csv_path),
        "--run-type",
        "MATH_ISOLATE",
        "--rates",
        str(rates),
        "--json",
    ]
    for s in select:
        argv += ["--select", s]
    return subprocess.run(argv, capture_output=True, text=True)


def test_cli_select_must_match_one_row(tmp_path):
    rows = [_csv_row(50.0, dest_acc="No"), _csv_row(50.0, dest_acc="Yes")]
    assert _cli(tmp_path, rows, ["mathop=Sigmoid"]).returncode == 2
    assert _cli(tmp_path, rows, ["mathop=Nope"]).returncode == 2
    ok = _cli(tmp_path, rows, ["mathop=Sigmoid", "dest_acc=No"])
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["bottleneck"] == "math"


def test_cli_ignores_non_tile_loop_markers(tmp_path):
    rows = [_csv_row(50.0), _csv_row(999.0, marker="INIT")]
    assert _cli(tmp_path, rows, ["mathop=Sigmoid"]).returncode == 0


def test_cli_exit_2_when_not_reconciled(tmp_path):
    result = _cli(tmp_path, [_csv_row(50.0)], ["mathop=Sigmoid"], cycles=7000)
    assert result.returncode == 2
    assert json.loads(result.stdout)["reconcile"]["ok"] is False


def test_bubbles_become_named_causes_and_idle_is_residual(tmp_path):
    thread = _thread(2560, idle=640)
    thread["bubbles"] = [
        {"after": "0x99", "before": "0x72", "count": 128, "cycles": 512}
    ]
    report = _report(6400, {"math": dict(thread, stall={"sfpu": 3200})})
    rates = perf_headroom.load_rates([_rates(tmp_path, "r.json", BASE_RATES)])
    names = {"0x99": "SFPNONLINEAR", "0x72": "SFPSTORE"}
    verdict = perf_headroom.headroom_verdict(
        report, _csv_row(50.0), rates, run_type="MATH_ISOLATE", opcode_names=names
    )
    causes = dict((name, value) for name, value in verdict["causes"])
    assert causes["math.bubble.SFPNONLINEAR->SFPSTORE"] == 4.0
    assert causes["math.idle"] == 1.0


def test_bottleneck_is_busiest_issuer_not_the_waiting_thread(tmp_path):
    # unpack mostly stalled waiting on math (instissue); math issues most of the window
    report = _report(
        6400,
        {
            "math": _thread(5760, stall={"sfpu": 128}, idle=512),
            "unpack": _thread(64, stall={"instissue": 6200}, idle=136),
        },
    )
    verdict = _verdict(tmp_path, report, _csv_row(50.0))
    assert verdict["bottleneck"] == "math"
    assert verdict["waiting_threads"] == {"unpack": {"instissue": 48.438}}


def test_missing_core_signal_is_refused(tmp_path):
    # I3: the wave tool lists signals without which stall time would read as idle
    report = _report(6400, {"math": _thread(2560, idle=3840)}, status="missing_signals")
    report["missing_signals"] = ["math.some_reason"]
    report["missing_core_signals"] = ["math.valid"]
    with pytest.raises(ValueError, match="missing core signals"):
        _verdict(tmp_path, report, _csv_row(50.0))


def test_missing_refinement_reason_is_accepted(tmp_path):
    report = _report(6400, {"math": _thread(2560, idle=3840)}, status="missing_signals")
    report["missing_signals"] = ["math.some_reason", "math.wait.some_condition"]
    report["missing_core_signals"] = []
    assert _verdict(tmp_path, report, _csv_row(50.0))["bottleneck"] == "math"
