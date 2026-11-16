# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import io
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "perf"))
import nightly_data_integrity as ndi

DAY = dt.date(2026, 11, 8)
HEAD = "tile_cnt,speed_of_light,marker,mean(L1_TO_L1),TEXT_SIZE(L1_TO_L1)\n"


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return zipfile.ZipFile(io.BytesIO(buf.getvalue()))


def _csv(sol, rows):
    return HEAD + "".join(f"{t},{sol},{m},{v},9\n" for t, m, v in rows)


def _rows(scale=1.0):
    return [(t, "KERNEL", 1000 * t * scale) for t in range(1, 13)]


def test_read_perf_finds_bad_values_and_duplicate_keys():
    text = _csv(False, [(1, "KERNEL", 5), (1, "KERNEL", 6), (2, "KERNEL", 0)])
    text += "3,False,KERNEL,abc,9\n4,False,KERNEL,,9\n"
    data, bad, dups = ndi.read_perf(
        [_zip({"r/perf_x/perf_x.csv": text, "r/perf_x/perf_x.post.csv": "junk"})]
    )
    assert len(data) == 4
    assert bad == {"perf_x": 2}
    assert dups == {"perf_x": 1}


def test_stuck_ignores_setup_markers_and_short_series():
    rows = [(t, m, 7) for t in range(12) for m in ("INIT", "KERNEL")]
    rows += [(t, "TILE_LOOP", 7) for t in range(3)]
    data, _, _ = ndi.read_perf([_zip({"perf_s.csv": _csv(False, rows)})])
    assert ndi.stuck(data) == {"perf_s": 1}


def test_below_sol_needs_both_slack_and_cycles():
    sol, _, _ = ndi.read_perf([_zip({"perf_m.csv": _csv(True, _rows())})])
    rows = [(1, "KERNEL", 900), (2, "KERNEL", 1800), (3, "KERNEL", 2990)]
    non_sol, _, _ = ndi.read_perf([_zip({"perf_m.csv": _csv(False, rows)})])
    # 1: 10 % but 100 cycles; 2: 10 % and 200 cycles; 3: under 5 %.
    assert ndi.below_sol(sol, non_sol) == {"perf_m": 1}


def test_baseline_check_names_every_shortfall():
    gate = {"baseline": "85b69f498da", "baseline_age": 19}
    problems, line = ndi.baseline_check(
        "SoL BH", gate, (83, 17), {"a"}, {"a", "b"}, {"a", "c"}
    )
    assert problems == [
        "SoL BH: 19 days old; 83% of points compared; "
        "1 module(s) without a baseline point (b); 1 baseline module(s) not measured (c)"
    ]
    assert "83 of 100 points compared (83%)" in line
    problems, _ = ndi.baseline_check(
        "SoL BH", {"baseline": "x", "baseline_age": 1}, (99, 1), {"a"}, {"a"}, None
    )
    assert problems == []


def test_gate_report_reads_counts_and_modules():
    zf = _zip(
        {
            "perf_gate_report.md": "- 378248 points compared, **1515 regressed point(s) "
            "in 1487 finding(s)**, 9 improved point(s) in 3 finding(s), 79179 new point(s)\n",
            "perf_gate_report.points.csv": "marker,test_module\nKERNEL,perf_a\nINIT,perf_b\n",
        }
    )
    assert ndi.gate_report(zf) == ((378248, 79179), {"perf_a", "perf_b"})


def test_check_end_to_end(monkeypatch):
    log = (
        "2026-11-08T03:39:45.4Z baseline baseline-20261020-39432951782-blackhole "
        "(85b69f498dac8ae354e722735ad26f03345e43c2, 2026-10-20 16:07:18+00:00): 1 rows\n"
        "verdict=regressed regressions=3 should_post=True\n"
    )
    report = _zip(
        {
            "perf_gate_report.md": "- 12 points compared, **3 regressed point(s) in 1 "
            "finding(s)**, 0 improved point(s) in 0 finding(s), 0 new point(s)\n",
            "perf_gate_report.points.csv": "marker,test_module\nKERNEL,perf_m\n",
        }
    )
    zips = {
        "sol-bh": _zip({"perf_m.csv": _csv(True, _rows())}),
        "non-bh": _zip({"perf_m.csv": _csv(False, _rows(0.5))}),
        "base-bh": _zip({"perf_m.csv": "", "perf_gone.csv": ""}),
        "report": report,
    }
    arts = {
        1: {
            "perf-data-blackhole-1": "sol-bh",
            "llk-perf-gate-report-blackhole": "report",
        },
        2: {
            "perf-data-blackhole-1": "non-bh",
            "llk-perf-gate-report-blackhole": "report",
        },
        39432951782: {"perf-data-blackhole-1": "base-bh"},
    }
    runs = iter([{"id": 1}, {"id": 2}])
    monkeypatch.setattr(ndi, "scheduled_run", lambda *a: next(runs))
    monkeypatch.setattr(
        ndi,
        "run_jobs",
        lambda repo, run_id: [
            {
                "id": run_id,
                "name": "Gate blackhole / Compare perf",
                "conclusion": "failure",
            }
        ],
    )
    monkeypatch.setattr(ndi, "job_log", lambda *a: log)
    monkeypatch.setattr(ndi, "artifacts", lambda repo, run_id: arts[int(run_id)])
    monkeypatch.setattr(ndi, "artifact_zip", lambda repo, i: zips[i])
    out = ndi.check("o/r", DAY)
    assert out["name"] == "Perf data integrity"
    assert out["problems"] == [
        "SoL BH: 19 days old; 1 baseline module(s) not measured (perf_gone)",
        "non-SoL BH: 19 days old; 1 baseline module(s) not measured (perf_gone)",
        "SoL WH: no perf data",
        "non-SoL WH: no perf data",
    ]
    assert "BH: 12 non-SoL point(s) more than 5% below SoL: perf_m 12" in out["lines"]


def test_failed_download_is_a_problem_not_a_crash(monkeypatch):
    calls = []

    def refuse(*a, **k):
        calls.append(a)
        return type("R", (), {"returncode": 1, "stderr": b"503 Egress"})()

    monkeypatch.setattr(ndi.subprocess, "run", refuse)
    monkeypatch.setattr(ndi.time, "sleep", lambda s: None)
    zips, failed = ndi.perf_zips("o/r", {"perf-data-wormhole-1": 7, "x": 8}, "wormhole")
    assert (zips, failed) == ([], ["perf-data-wormhole-1"])
    assert len(calls) == 2
