# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measurement arithmetic of the prefill-matrix producer, kept free of ttnn/torch so it is unit-testable on any host
(models/demos/minimax_m3/tests/unit/test_prefill_matrix_math.py): reading the ranks' per-chunk timing CSVs, the
steady-state window that excludes pipeline fill/drain, and the per-request TTFT-under-load statistics."""

import csv
import math
import os
import statistics
import time


def read_rank_csv(timing_dir: str, rank: int) -> dict:
    """{c: (compute_start_epoch_s, compute_ms)} for one rank; re-opened every call (NFS close-to-open).
    The file is appended from another host while we read it: only newline-terminated lines are accepted, so a
    torn last line ("450.1" of "450.123") is never taken as a finished chunk."""
    out = {}
    path = os.path.join(timing_dir, f"rank{rank}.csv")
    try:
        with open(path) as fh:
            data = fh.read()
    except FileNotFoundError:
        return out
    lines = data.split("\n")
    if not data.endswith("\n"):
        lines.pop()  # partial write in flight
    for row in csv.reader(lines):
        if len(row) == 4:
            out[int(row[1])] = (float(row[2]), float(row[3]))
    return out


def wait_for_chunk(timing_dir: str, rank: int, c: int, timeout_s: float, poll_s: float = 0.2) -> tuple:
    """Poll rank ``rank``'s CSV until chunk ``c`` has a row; returns (compute_start, compute_ms)."""
    t0 = time.perf_counter()
    while True:
        rows = read_rank_csv(timing_dir, rank)
        if c in rows:
            return rows[c]
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"rank {rank} chunk c={c} not done after {timeout_s:.0f} s (have {len(rows)} rows)")
        time.sleep(poll_s)


def steady_state(ends: dict, c0: int, c_end: int, stages: int, new_tokens: dict, chunk: int):
    """Steady-state throughput of a back-to-back chunk stream from the LAST rank's chunk end times.

    ``ends`` maps chunk index -> wall-clock end time (s) for every chunk in [c0, c_end]; ``new_tokens`` maps chunk
    index -> new tokens carried by that chunk (``chunk`` for full chunks, less for a ragged last chunk). The first
    and last ``stages`` chunks are dropped (pipeline fill / drain); the window is the time from the end of the
    chunk before the first kept one to the end of the last kept one, so it spans exactly ``mid`` chunk periods.
    Returns None when fewer than 4 chunks remain."""
    lo, hi = c0 + stages, c_end - stages
    mid = hi - lo + 1
    if mid < 4:
        return None
    window = ends[hi] - ends[lo - 1]
    periods = [ends[c] - ends[c - 1] for c in range(lo, hi + 1)]
    return {
        "mid_chunks": mid,
        "window_s": window,
        "steady_new_tps": sum(new_tokens[c] for c in range(lo, hi + 1)) / window,
        "steady_processed_tps": mid * chunk / window,
        "chunk_period_ms_median": statistics.median(periods) * 1000.0,
    }


def ttft_stats(req: dict, ends: dict, c_full: int = 0) -> dict:
    """Per-request TTFT under load (ms): push of the request's first chunk -> last-rank end of its last chunk.
    Every request must have its last chunk in ``ends`` (the caller asserted the stream is complete). Requests
    whose first chunk index is below ``c_full`` entered a still-filling pipeline and are excluded (all requests
    are used if that leaves none). p90 is the nearest-rank percentile."""
    missing = [k for k, r in req.items() if r["c_last"] not in ends]
    assert not missing, f"{len(missing)} request(s) have no completion row: {missing[:4]}"
    kept = [r for r in req.values() if r["c_first"] >= c_full] or list(req.values())
    ttfts = sorted((ends[r["c_last"]] - r["t_push_first"]) * 1000.0 for r in kept)
    return {
        "ttft_under_load_requests": len(ttfts),
        "ttft_under_load_ms_median": statistics.median(ttfts),
        "ttft_under_load_ms_p90": ttfts[math.ceil(0.9 * len(ttfts)) - 1],
        "ttft_under_load_ms_min": ttfts[0],
        "ttft_under_load_ms_max": ttfts[-1],
    }
