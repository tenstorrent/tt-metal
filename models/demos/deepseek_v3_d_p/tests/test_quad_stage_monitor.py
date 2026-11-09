# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side logic of the quad screen's per-stage check; no device needed."""

import json

import torch

from models.demos.deepseek_v3_d_p.tt.runners.quad_stage_monitor import StageMonitor


def _shards(seed, n=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 1, 32, 64, generator=g).to(torch.bfloat16) for _ in range(n)]


def _monitor(tmp_path, rank=1, mode="block_random"):
    return StageMonitor(rank=rank, mode=mode, results_dir=tmp_path, mesh_cols=4, fabric="2d_torus_xy", host="hx")


def _run(mon, iters, chunks=2, out_override=None):
    for it in range(iters):
        for c in range(chunks):
            mon.check("input", c, _shards(10 + c))
            out = _shards(20 + c)
            if out_override is not None:
                out = out_override(it, c, out)
            mon.check("output", c, out)


def _doc(tmp_path, rank=1):
    return json.loads((tmp_path / f"rank{rank}.json").read_text())


def test_baseline_then_equal_passes(tmp_path):
    mon = _monitor(tmp_path)
    _run(mon, 3)
    mon.finish()
    doc = _doc(tmp_path)
    assert doc["first_bad"] is None and doc["error"] is None and doc["done"] is True
    assert doc["iterations"] == 3 and doc["chunks_done"] == 6
    assert doc["last_in"] == [2, 1] and doc["last_out"] == [2, 1]
    assert doc["host"] == "hx" and doc["rank"] == 1 and doc["fabric"] == "2d_torus_xy"


def test_mismatch_records_device_coords(tmp_path):
    def corrupt(it, c, out):
        if it == 2 and c == 1:
            out[6] = out[6].clone()
            out[6][0, 0, 0, 0] += 1
        return out

    mon = _monitor(tmp_path)
    _run(mon, 3, out_override=corrupt)
    doc = _doc(tmp_path)
    assert doc["first_bad"] == {"iter": 2, "chunk": 1, "what": "output", "devices": [[1, 2]], "n_diff": 1}
    assert doc["mismatches"] == [doc["first_bad"]]


def test_expected_pattern_mode(tmp_path):
    mon = _monitor(tmp_path, mode="passthrough")
    expected = _shards(5)
    got = [s.clone() for s in expected]
    got[0][0, 0, 1, 1] = 7
    mon.check("input", 0, got, expected=expected)
    doc = _doc(tmp_path)
    assert doc["first_bad"]["iter"] == 0 and doc["first_bad"]["devices"] == [[0, 0]]


def test_vacuous_baseline_fails(tmp_path):
    mon = _monitor(tmp_path)
    mon.check("input", 0, _shards(1))
    mon.check("output", 0, [torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)] * 8)
    doc = _doc(tmp_path)
    assert "iteration 0 output chunk 0" in doc["error"] and "all zeros" in doc["error"]


def test_partial_last_iteration_ok(tmp_path):
    mon = _monitor(tmp_path)
    _run(mon, 2)
    mon.check("input", 0, _shards(10))
    mon.check("output", 0, _shards(20))
    mon.finish()
    doc = _doc(tmp_path)
    assert doc["first_bad"] is None and doc["error"] is None
    assert doc["iterations"] == 3 and doc["chunks_done"] == 5


def test_hash_line_written_per_chunk(tmp_path):
    mon = _monitor(tmp_path)
    _run(mon, 2)
    lines = [json.loads(l) for l in (tmp_path / "rank1.hashes.jsonl").read_text().splitlines()]
    assert [(h["iter"], h["chunk"]) for h in lines] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(lines[0]["in"]) == 8 and len(lines[0]["out"]) == 8
    assert lines[0]["in"] == lines[2]["in"] and lines[0]["in"] != lines[1]["in"]


def test_mismatch_list_capped_at_20(tmp_path):
    def corrupt(it, c, out):
        if it > 0:
            out[0] = out[0] + 1
        return out

    mon = _monitor(tmp_path)
    _run(mon, 15, out_override=corrupt)
    doc = _doc(tmp_path)
    assert len(doc["mismatches"]) == 20
    assert doc["first_bad"]["iter"] == 1


def test_inject_flips_one_device(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_QUAD_INJECT", "1:output:1")
    mon = _monitor(tmp_path)
    _run(mon, 2)
    doc = _doc(tmp_path)
    assert doc["first_bad"]["what"] == "output" and doc["first_bad"]["iter"] == 1
    assert doc["first_bad"]["devices"] == [[0, 0]]


def test_inject_other_rank_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_QUAD_INJECT", "2:output:1")
    mon = _monitor(tmp_path)
    _run(mon, 2)
    assert _doc(tmp_path)["first_bad"] is None


def test_fail_records_error(tmp_path):
    mon = _monitor(tmp_path)
    mon.fail("RuntimeError: boom")
    assert _doc(tmp_path)["error"] == "RuntimeError: boom"


def test_one_dead_chip_in_baseline_fails(tmp_path):
    mon = _monitor(tmp_path)
    mon.check("input", 0, _shards(1))
    out = _shards(2)
    out[5] = torch.zeros_like(out[5])
    mon.check("output", 0, out)
    doc = _doc(tmp_path)
    assert "all zeros" in doc["error"] and "(1,1)" in doc["error"]
