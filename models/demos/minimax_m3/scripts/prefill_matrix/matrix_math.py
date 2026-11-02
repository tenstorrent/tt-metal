# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measurement arithmetic of the prefill-matrix producer, kept free of ttnn/torch so it is unit-testable on any host
(models/demos/minimax_m3/tests/unit/test_prefill_matrix_math.py): chunk completion from the runner's layer acks, the
steady-state window that excludes pipeline fill/drain, and the per-request TTFT-under-load statistics."""

import math
import statistics
import threading
from collections import deque


class AckLedger:
    """Per-chunk first and last layer-ack times on the producer's clock, keyed by the producer's chunk index.

    The runner acks (slot, pos_start, pos_end, layer span), and the matrix pushes the same positions again while
    earlier copies are still in flight. Each rank acks its layers in push order, so an ack belongs to the oldest
    in-flight copy of its key that has not seen that layer span yet. A chunk is done once its spans cover
    ``chunk_layers`` layers."""

    def __init__(self, chunk_layers: int):
        self._chunk_layers = chunk_layers
        self._open = {}  # key -> deque of [c, span starts seen, layers covered]
        self._cond = threading.Condition()
        self.first = {}
        self.ends = {}
        self.stray = 0

    def push(self, c: int, key: tuple) -> None:
        with self._cond:
            self._open.setdefault(key, deque()).append([c, set(), 0])

    def ack(self, key: tuple, layer_start: int, layer_end: int, ts: float) -> None:
        with self._cond:
            entry = next((e for e in self._open.get(key, ()) if layer_start not in e[1]), None)
            if entry is None:
                self.stray += 1
                return
            c, seen, _ = entry
            seen.add(layer_start)
            entry[2] += layer_end - layer_start
            self.first.setdefault(c, ts)
            if entry[2] >= self._chunk_layers:
                self.ends[c] = ts
                self._open[key].remove(entry)
                self._cond.notify_all()

    def wait(self, c: int, timeout_s: float) -> float:
        """Last-ack time of chunk ``c``."""
        with self._cond:
            if not self._cond.wait_for(lambda: c in self.ends, timeout=timeout_s):
                raise TimeoutError(f"chunk c={c} not fully acked after {timeout_s:.0f} s ({len(self.ends)} chunks done)")
            return self.ends[c]


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
    """Per-request TTFT under load (ms): push of the request's first chunk -> last layer ack of its last chunk.
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
