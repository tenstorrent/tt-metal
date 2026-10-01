# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""parse_pipeline_perf on a synthetic two-rank runner log with known gaps, request bounds and a dropped line."""

import importlib.util
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "parse_pipeline_perf.py"
_spec = importlib.util.spec_from_file_location("parse_pipeline_perf", _SRC)
ppp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppp)

CHUNK = 100
# Two slots in round-robin, three chunks each. Rank 0 gaps: 0.5 0.5 0.4 0.4 0.4 -> steady 400 ms after the two fill
# gaps and the drain gap are dropped. Rank 1 gaps: 0.5 0.5 0.45 0.45 0.42 -> steady 450 ms, the bottleneck.
RANK0_T = [0.0, 0.5, 1.0, 1.4, 1.8, 2.2]
RANK1_T = [0.1, 0.6, 1.1, 1.55, 2.0, 2.42]
SLOT = [0, 1, 0, 1, 0, 1]
START = [0, 0, CHUNK, CHUNK, 2 * CHUNK, 2 * CHUNK]


def _line(rank, c, t):
    return (
        f"[1,{rank}]<stderr>: ... | INFO | [pp rank {rank}] CHUNK_START c={c} compute_start={t:.6f} "
        f"slot={SLOT[c]} [{START[c]},{START[c] + CHUNK}) provided=None\n"
    )


def _write_log(tmp_path, drop=()):
    lines = [_line(0, c, t) for c, t in enumerate(RANK0_T)]
    lines += [_line(1, c, t) for c, t in enumerate(RANK1_T) if (1, c) not in drop]
    path = tmp_path / "runner.log"
    path.write_text("noise\n" + "".join(lines))
    return str(path)


def test_steady_state_and_latency(tmp_path):
    s = ppp.summarize([_write_log(tmp_path)])
    assert s["n_events"] == 12 and s["chunk_tokens"] == CHUNK
    assert s["per_rank"][0]["median_gap_ms"] == pytest.approx(400.0)
    assert s["per_rank"][1]["median_gap_ms"] == pytest.approx(450.0)
    t = s["throughput"]
    assert t["bottleneck_rank"] == 1
    assert t["steady_tok_s"] == pytest.approx(CHUNK / 0.45)
    assert t["ceiling_tok_s"] == pytest.approx(CHUNK / 0.4)
    # Request opened at c=0 ends with rank 1's c=4 (2.0 - 0.0); the one opened at c=1 with its c=5 (2.42 - 0.5).
    assert s["n_requests"] == 2
    assert s["latency_p50_s"] == pytest.approx((2.0 + 1.92) / 2)
    assert s["latency_p90_s"] == pytest.approx(2.0)


def test_chunk_size_override_scales_throughput(tmp_path):
    s = ppp.summarize([_write_log(tmp_path)], chunk_tokens=2 * CHUNK)
    assert s["throughput"]["steady_tok_s"] == pytest.approx(2 * CHUNK / 0.45)


def test_dropped_line_does_not_shift_request_pairing(tmp_path):
    s = ppp.summarize([_write_log(tmp_path, drop={(1, 3)})])
    assert s["n_requests"] == 2
    assert s["latency_p90_s"] == pytest.approx(2.0)


def test_unparseable_log_is_an_error(tmp_path, expect_error):
    path = tmp_path / "empty.log"
    path.write_text("no chunk lines here\n")
    with expect_error(ValueError, "no CHUNK_START lines"):
        ppp.summarize([str(path)])
    with expect_error(SystemExit, "no CHUNK_START lines"):
        ppp.main([str(path)])
