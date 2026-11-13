# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for perf_headroom.py — wave breakdown + perf CSV -> headroom verdict."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parent / "perf_headroom.py"
_SPEC = importlib.util.spec_from_file_location("perf_headroom", _PATH)
perf_headroom = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_headroom)

_ASSEMBLY_YAML = (
    Path(__file__).resolve().parents[2]
    / "tt_llk_quasar"
    / "instructions"
    / "assembly.yaml"
)

# The default fixture: 32 loops x 4 tiles = 128 tile iterations in a 6400-cycle window.
LOOP_FACTOR = 32
TILE_CNT = 4
TILES = LOOP_FACTOR * TILE_CNT
WINDOW_CYCLES = 6400
PER_TILE = WINDOW_CYCLES / TILES  # 50.0


def _rates(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


BASE_RATES = {
    "default": {"cycles": 1, "source": "single-issue per thread"},
    "opcodes": {"SFPMAD": {"cycles": 1, "source": "assembly.yaml SFPMAD"}},
}


def _base_rates(tmp_path: Path) -> dict:
    return perf_headroom.load_rates([_rates(tmp_path, "r.json", BASE_RATES)])


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
        "loop_factor": str(LOOP_FACTOR),
        "tile_cnt": str(TILE_CNT),
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
    thread = _thread(6, by_opcode={"SFPMAD": 4, "FOO": 2})
    assert perf_headroom.unit_floor_cycles(thread, _base_rates(tmp_path)) == (
        6,
        ["FOO"],
    )


def test_unit_floor_prices_opcodes_from_the_rate_table(tmp_path):
    # SFPSWAP at 3 cycles: 2 x 3 + 4 x 1 (SFPMAD) + 2 x 1 (default) = 12
    rates = perf_headroom.load_rates(
        [
            _rates(tmp_path, "a.json", BASE_RATES),
            _rates(
                tmp_path,
                "b.json",
                {"opcodes": {"SFPSWAP": {"cycles": 3, "source": "s"}}},
            ),
        ]
    )
    thread = _thread(8, by_opcode={"SFPSWAP": 2, "SFPMAD": 4, "FOO": 2})
    assert perf_headroom.unit_floor_cycles(thread, rates) == (12, ["FOO"])


def test_unit_floor_without_opcodes_defaults_everything(tmp_path):
    assert perf_headroom.unit_floor_cycles(_thread(7), _base_rates(tmp_path)) == (
        7,
        ["<all>"],
    )


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


def test_opcode_headers_may_carry_a_trailing_comment(tmp_path):
    yaml_text = (
        "UNPACR_NOP: # Misc unpack instruction\n    op_binary: 0x43\n"
        "RESOURCEDECL: # fixme: fcov\n    op_binary: 0x05\n"
    )
    path = tmp_path / "assembly.yaml"
    path.write_text(yaml_text)
    assert perf_headroom.load_opcode_names(path) == {
        "0x43": "UNPACR_NOP",
        "0x05": "RESOURCEDECL",
    }


@pytest.mark.skipif(not _ASSEMBLY_YAML.is_file(), reason="no Quasar assembly.yaml")
def test_every_quasar_op_binary_gets_its_mnemonic():
    text = _ASSEMBLY_YAML.read_text()
    headers = re.findall(r"^([A-Z][A-Z0-9_]*):", text, flags=re.M)
    op_binaries = re.findall(r"^\s+op_binary:", text, flags=re.M)
    assert len(headers) == len(op_binaries)
    table = perf_headroom.load_opcode_names(_ASSEMBLY_YAML)
    named = {n for value in table.values() for n in value.split("|")}
    assert named == set(headers)
    assert table["0x43"] == "UNPACR_NOP"


def test_per_tile_matches_post_csv_formula():
    assert perf_headroom.per_tile(WINDOW_CYCLES, LOOP_FACTOR, TILE_CNT) == PER_TILE


# --- verdict -----------------------------------------------------------------


def _verdict(tmp_path, report, row, run_type="MATH_ISOLATE", **kw):
    return perf_headroom.headroom_verdict(
        report, row, _base_rates(tmp_path), run_type=run_type, **kw
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
    report = _report(6400, {"math": _thread(2560, idle=3840)})
    verdict = _verdict(
        tmp_path, report, _csv_row(50.0), best_count=15, best_count_source="hand count"
    )
    assert verdict["instruction_headroom"] == {
        "current": 20.0,
        "best": 15,
        "source": "hand count",
        "pct": 25.0,
    }


@pytest.mark.parametrize(
    "cycles, ok",
    [
        (WINDOW_CYCLES, True),
        (6528, True),  # +2.0 %: on the tolerance
        (6592, False),  # +3 %
        (5760, False),  # -10 %: the tolerance is on |delta|
    ],
)
def test_reconcile_tolerance_is_two_percent_either_way(tmp_path, cycles, ok):
    report = _report(cycles, {"math": _thread(100, idle=cycles - 100)})
    verdict = _verdict(tmp_path, report, _csv_row(PER_TILE))
    assert verdict["reconcile"]["ok"] is ok


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


def _cli(tmp_path, csv_rows, select, cycles=WINDOW_CYCLES, as_json=True, extra=()):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(_report(cycles, {"math": _thread(2560, idle=cycles - 2560)}))
    )
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
        *(["--json"] if as_json else []),
        *extra,
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


def test_cli_text_summary(tmp_path):
    extra = ["--best-count", "15", "--best-count-source", "hand count"]
    ok = _cli(
        tmp_path, [_csv_row(PER_TILE)], ["mathop=Sigmoid"], as_json=False, extra=extra
    )
    assert ok.returncode == 0, ok.stderr
    assert "-> OK" in ok.stdout
    assert "instr headroom  : 25.0 % (20.0 -> 15 instr/tile; hand count)" in ok.stdout
    bad = _cli(
        tmp_path, [_csv_row(PER_TILE)], ["mathop=Sigmoid"], cycles=7000, as_json=False
    )
    assert bad.returncode == 2
    assert "NOT RECONCILED" in bad.stdout


def test_cli_explicit_missing_assembly_yaml_is_an_error(tmp_path):
    extra = ["--assembly-yaml", str(tmp_path / "nope.yaml")]
    result = _cli(tmp_path, [_csv_row(PER_TILE)], ["mathop=Sigmoid"], extra=extra)
    assert result.returncode == 2
    assert "nope.yaml" in result.stderr


def test_bubbles_become_named_causes_and_idle_is_residual(tmp_path):
    thread = _thread(2560, idle=640)
    thread["bubbles"] = [
        {"after": "0x99", "before": "0x72", "count": 128, "cycles": 512}
    ]
    report = _report(6400, {"math": dict(thread, stall={"sfpu": 3200})})
    names = {"0x99": "SFPNONLINEAR", "0x72": "SFPSTORE"}
    verdict = _verdict(tmp_path, report, _csv_row(50.0), opcode_names=names)
    causes = dict((name, value) for name, value in verdict["causes"])
    assert causes["math.bubble.SFPNONLINEAR->SFPSTORE"] == 4.0
    assert causes["math.idle"] == 1.0


def test_repeated_bubble_pairs_accumulate(tmp_path):
    thread = _thread(2560, idle=3840)
    thread["bubbles"] = [
        {"after": "0x99", "before": "0x72", "count": 64, "cycles": 256},
        {"after": "0x99", "before": "0x72", "count": 64, "cycles": 256},
    ]
    names = {"0x99": "SFPNONLINEAR", "0x72": "SFPSTORE"}
    verdict = _verdict(
        tmp_path, _report(6400, {"math": thread}), _csv_row(50.0), opcode_names=names
    )
    causes = dict((name, value) for name, value in verdict["causes"])
    assert causes["math.bubble.SFPNONLINEAR->SFPSTORE"] == 4.0
    assert causes["math.idle"] == 26.0


def test_bottleneck_is_busiest_issuer_not_the_waiting_thread(tmp_path):
    # unpack mostly stalled waiting on math (instissue); math issues most of the window
    report = _report(
        6400,
        {
            "math": _thread(5760, stall={"sfpu": 128}, idle=512),
            "unpack": _thread(64, stall={"instissue": 6200}, idle=136),
        },
    )
    verdict = _verdict(tmp_path, report, _csv_row(50.0), run_type="L1_TO_L1")
    assert verdict["bottleneck"] == "math"
    assert verdict["waiting_threads"] == {"unpack": {"instissue": 48.438}}


def test_isolate_run_type_pins_the_bottleneck_to_the_isolated_thread(tmp_path):
    # UNPACK_ISOLATE: mocked math issues STALLWAIT + CLEARDVALID per tile and out-issues unpack.
    report = _report(
        6400,
        {
            "math": _thread(256, idle=6144),
            "unpack": _thread(192, stall={"unpack": 5760}, idle=448),
        },
    )
    row = _csv_row(50.0, **{"mean(UNPACK_ISOLATE)": "50.0"})
    verdict = _verdict(tmp_path, report, row, run_type="UNPACK_ISOLATE")
    assert verdict["bottleneck"] == "unpack"
    assert verdict["bound"]["kind"] == "issue"
    assert verdict["causes"][0] == ["unpack.unpack", 45.0]


def test_isolate_run_type_without_its_thread_is_refused(tmp_path):
    report = _report(6400, {"math": _thread(2560, idle=3840)})
    row = _csv_row(50.0, **{"mean(PACK_ISOLATE)": "50.0"})
    with pytest.raises(ValueError, match="isolates 'pack'"):
        _verdict(tmp_path, report, row, run_type="PACK_ISOLATE")


def test_thread_stalled_on_its_own_unit_is_not_waiting(tmp_path):
    # pack held by its busy packer is working; only its cross-thread stalls would make it wait.
    report = _report(
        6400,
        {
            "math": _thread(3200, idle=3200),
            "pack": _thread(640, stall={"pack": 5120, "instissue": 128}, idle=512),
        },
    )
    verdict = _verdict(tmp_path, report, _csv_row(50.0), run_type="L1_TO_L1")
    assert verdict["waiting_threads"] == {}


def test_breakdown_that_does_not_cover_the_window_is_refused(tmp_path):
    report = _report(6400, {"math": _thread(2560, idle=3000)})
    with pytest.raises(ValueError, match="does not cover the window"):
        _verdict(tmp_path, report, _csv_row(50.0))


def test_unknown_cycles_are_refused(tmp_path):
    thread = dict(_thread(2560, idle=3740), unknown=100)
    with pytest.raises(ValueError, match="100 unknown"):
        _verdict(tmp_path, _report(6400, {"math": thread}), _csv_row(50.0))


def test_missing_core_signal_is_refused(tmp_path):
    # the wave tool lists the signals without which stall time would read as idle
    report = _report(6400, {"math": _thread(2560, idle=3840)}, status="missing_signals")
    report["missing_signals"] = ["math.some_reason"]
    report["missing_core_signals"] = ["math.valid"]
    with pytest.raises(ValueError, match="missing core signals"):
        _verdict(tmp_path, report, _csv_row(50.0))


def test_missing_signals_without_the_core_key_is_refused(tmp_path):
    report = _report(6400, {"math": _thread(2560, idle=3840)}, status="missing_signals")
    report["missing_signals"] = ["math.some_reason"]
    with pytest.raises(ValueError, match="missing_core_signals"):
        _verdict(tmp_path, report, _csv_row(50.0))


def test_missing_refinement_reason_is_accepted(tmp_path):
    report = _report(6400, {"math": _thread(2560, idle=3840)}, status="missing_signals")
    report["missing_signals"] = ["math.some_reason", "math.wait.some_condition"]
    report["missing_core_signals"] = []
    assert _verdict(tmp_path, report, _csv_row(50.0))["bottleneck"] == "math"


def test_floor_above_measured_is_flagged_with_the_multi_cycle_opcodes(tmp_path):
    # 40 SWAP + 40 NOP + 48 other per tile issue in 128 cycles; a 2-cycle SWAP cost puts the floor at 168.
    rates = {
        "default": {"cycles": 1, "source": "s"},
        "opcodes": {"SFPSWAP": {"cycles": 2, "source": "s"}},
    }
    by_op = {"SFPSWAP": 40 * 128, "SFPNOP": 40 * 128, "SFPLOAD": 48 * 128}
    report = _report(128 * 128, {"math": _thread(128 * 128, by_opcode=by_op)})
    verdict = perf_headroom.headroom_verdict(
        report, _csv_row(128.0), rates, run_type="MATH_ISOLATE"
    )
    assert verdict["stall_headroom_pct"] < 0
    assert verdict["floor_check"] == {
        "ok": False,
        "multi_cycle_opcodes": ["SFPSWAP"],
    }
    assert "floor above measured" in perf_headroom._summary(verdict)


def test_floor_check_is_ok_normally(tmp_path):
    report = _report(6400, {"math": _thread(2560, idle=3840)})
    assert _verdict(tmp_path, report, _csv_row(50.0))["floor_check"]["ok"] is True


# ── what limits the kernel: only issue-bound kernels get a stall-headroom verdict ──


def _bound_verdict(threads, per_tile, run_type="L1_TO_L1"):
    tiles = 128
    report = _report(int(per_tile * tiles), threads)
    return perf_headroom.headroom_verdict(
        report,
        {
            "loop_factor": str(LOOP_FACTOR),
            "tile_cnt": str(TILE_CNT),
            f"mean({run_type})": str(per_tile),
        },
        {"default": {"cycles": 1, "source": "s"}, "opcodes": {}},
        run_type=run_type,
    )


def test_pack_bound_kernel_is_not_judged_on_the_issue_floor():
    # pack_untilize: math issues 12/tile, pack issues 9 and is held 43 by its busy packer.
    t = 128
    v = _bound_verdict(
        {
            "math": _thread(12 * t, stall={"instissue": 1 * t}, idle=40 * t),
            "pack": _thread(9 * t, stall={"instissue": 43 * t}, idle=1 * t),
            "unpack": _thread(1 * t, idle=52 * t),
        },
        53.0,
    )
    assert v["bound"] == {"kind": "unit", "thread": "pack", "occupancy_pct": 98.113}
    assert v["stall_headroom_pct"] is None
    assert v["occupancy_pct"]["math"] == 24.528
    summary = perf_headroom._summary(v)
    assert "unit-bound (pack" in summary and "not judged" in summary


def test_issue_saturated_math_stays_the_bottleneck_even_if_unpack_waits_longer():
    # sigmoid L1_TO_L1: math issues 230/272.8, unpack is held 254 waiting on math.
    t = 128
    v = _bound_verdict(
        {
            "math": _thread(230 * t, stall={"sfpu": 4 * t}, idle=39 * t),
            "unpack": _thread(1 * t, stall={"instissue": 254 * t}, idle=18 * t),
        },
        273.0,
    )
    assert v["bound"]["kind"] == "issue" and v["bound"]["thread"] == "math"
    assert v["stall_headroom_pct"] == round(100 * (273 - 230) / 273, 3)


def test_isolated_math_run_is_issue_bound_even_with_long_gaps():
    # ReduceRow-like MATH_ISOLATE: math issues only 37 % of the time; the rest are its own gaps.
    t = 128
    v = _bound_verdict(
        {"math": _thread(37 * t, idle=63 * t)}, 100.0, run_type="MATH_ISOLATE"
    )
    assert v["bound"]["kind"] == "issue"
    assert v["stall_headroom_pct"] == 63.0


def test_no_thread_busy_most_of_the_time_is_sync_bound():
    t = 128
    v = _bound_verdict(
        {
            "math": _thread(10 * t, idle=30 * t),
            "pack": _thread(4 * t, stall={"instissue": 11 * t}, idle=25 * t),
        },
        40.0,
    )
    assert v["bound"]["kind"] == "sync"
    assert v["stall_headroom_pct"] is None
