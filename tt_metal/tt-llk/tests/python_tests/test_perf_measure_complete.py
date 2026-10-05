# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate's per-arch measurement check (lives in tt_metal/tt-llk/perf)."""

import json
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from measure_complete import legs_of, main, measure_complete


def _job(name, conclusion="success"):
    return {"name": name, "conclusion": conclusion}


_NIGHT = [
    _job("LLK perf tests / load-test-matrix"),
    _job("LLK perf tests / compile-quasar-perf / Quasar perf compile"),
    *[
        _job(f"LLK perf tests / llk_perf_wormhole group {i}/5 [wh_n150_civ2]")
        for i in range(1, 6)
    ],
    _job("LLK perf tests / llk_perf_blackhole group 1/5 [bh_p150b_civ2]", "failure"),
    *[
        _job(f"LLK perf tests / llk_perf_blackhole group {i}/5 [bh_p150b_civ2]")
        for i in range(2, 6)
    ],
    _job("Gate wormhole (speed of light) / Compare perf: current vs main", None),
]


def test_a_lost_leg_on_the_other_arch_does_not_skip_this_one():
    complete, reason = measure_complete("failure", _NIGHT, "wormhole")
    assert complete and "all 5 wormhole legs passed" in reason


def test_a_lost_leg_on_this_arch_is_incomplete():
    complete, reason = measure_complete("failure", _NIGHT, "blackhole")
    assert not complete
    assert "1 of 5 blackhole legs" in reason and "group 1/5" in reason


def test_legs_of_matches_every_suite_and_only_its_arch():
    jobs = [
        _job(
            "Measure the PR head on blackhole / llk_perf_blackhole group 1/5 [bh_p150b_civ2]"
        ),
        _job(
            "Measure the branch point on blackhole / llk_perf_blackhole group 1/5 [bh_p150b_civ2]"
        ),
        _job(
            "LLK perf merge gate (measure) / llk_perf_merge_gate_blackhole group 2/2 [bh_p150b_civ2_viommu]"
        ),
        _job("LLK perf tests / llk_perf_wormhole group 1/5 [wh_n150_civ2]"),
        _job("Gate blackhole (speed of light) / Compare perf: current vs main"),
    ]
    assert len(legs_of(jobs, "blackhole")) == 3
    assert len(legs_of(jobs, "wormhole")) == 1


def test_a_failed_leg_of_the_branch_point_counts_too():
    jobs = [
        _job(
            "Measure the PR head on wormhole / llk_perf_wormhole group 1/5 [wh_n150_civ2]"
        ),
        _job(
            "Measure the branch point on wormhole / llk_perf_wormhole group 1/5 [wh_n150_civ2]",
            "cancelled",
        ),
    ]
    assert not measure_complete("success,cancelled", jobs, "wormhole")[0]


def test_without_legs_the_job_results_decide():
    assert measure_complete("success,skipped", [], "blackhole")[0]
    assert measure_complete("", None, "blackhole")[0]
    assert not measure_complete("success,failure", None, "blackhole")[0]


def test_main_reads_the_jobs_file(tmp_path, capsys):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("".join(json.dumps(j) + "\n" for j in _NIGHT))
    main(["--arch", "wormhole", "--results", "failure", "--jobs", str(jobs)])
    assert "complete=true" in capsys.readouterr().out


def test_main_falls_back_to_the_results_when_the_jobs_cannot_be_read(tmp_path, capsys):
    main(
        [
            "--arch",
            "wormhole",
            "--results",
            "failure",
            "--jobs",
            str(tmp_path / "missing.jsonl"),
        ]
    )
    out = capsys.readouterr().out
    assert "complete=false" in out and "::warning::" in out


def test_a_skipped_leg_is_incomplete():
    """A leg that did not run measured nothing, so its arch is not complete."""
    jobs = [
        _job("LLK perf tests / llk_perf_wormhole group 1/5 [wh_n150_civ2]", "skipped")
    ]
    assert not measure_complete("success", jobs, "wormhole")[0]


def test_main_prints_how_many_legs_it_found(tmp_path, capsys):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("".join(json.dumps(j) + "\n" for j in _NIGHT))
    main(["--arch", "blackhole", "--jobs", str(jobs)])
    assert "legs=5" in capsys.readouterr().out
