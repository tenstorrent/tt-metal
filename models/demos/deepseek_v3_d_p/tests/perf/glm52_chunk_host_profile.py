# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, single-chunk host timing for the GLM-5.2 eager prefill test.

FastOperation calls are timed on the caller thread. The time between them is
reported by operation pair and next model call site. These are diagnostic
measurements: the Python wrapper itself adds overhead, so use the ordinary
unprofiled chunk timing for absolute performance comparisons.
"""

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import patch

import ttnn


def _model_caller():
    frame = sys._getframe(2)
    while frame:
        filename = frame.f_code.co_filename
        if "/models/demos/deepseek_v3_d_p/tt/" in filename:
            return f"{filename.rsplit('/', 1)[-1]}:{frame.f_lineno}:{frame.f_code.co_name}"
        frame = frame.f_back
    return "outside_model"


class ChunkHostProfile:
    def __init__(self):
        self.calls = []
        self.forward_start_ns = 0
        self.forward_end_ns = 0
        self.sync_ns = 0

    @contextmanager
    def capture_forward(self):
        original_call = ttnn.decorators.FastOperation.__call__
        depth = 0

        def timed_call(op, *args, **kwargs):
            nonlocal depth
            if depth:
                return original_call(op, *args, **kwargs)
            depth = 1
            caller = _model_caller()
            start = time.perf_counter_ns()
            try:
                return original_call(op, *args, **kwargs)
            finally:
                end = time.perf_counter_ns()
                self.calls.append((op.python_fully_qualified_name, caller, start, end))
                depth = 0

        with patch.object(ttnn.decorators.FastOperation, "__call__", timed_call):
            self.forward_start_ns = time.perf_counter_ns()
            try:
                yield
            finally:
                self.forward_end_ns = time.perf_counter_ns()

    def report(self, iteration, chunk):
        forward_ns = self.forward_end_ns - self.forward_start_ns
        op_ns = sum(end - start for _, _, start, end in self.calls)
        gaps = []
        previous_end = self.forward_start_ns
        previous_name = "forward_start"
        for name, caller, start, end in self.calls:
            gaps.append((previous_name, name, caller, max(0, start - previous_end)))
            previous_end, previous_name = end, name
        gaps.append((previous_name, "forward_end", "forward_return", max(0, self.forward_end_ns - previous_end)))
        gap_ns = sum(gap for _, _, _, gap in gaps)

        def ranked(rows, limit=25):
            return sorted(rows, key=lambda row: row[1], reverse=True)[:limit]

        by_op = defaultdict(lambda: [0, 0])
        for name, _, start, end in self.calls:
            by_op[name][0] += end - start
            by_op[name][1] += 1
        by_gap = defaultdict(lambda: [0, 0])
        for previous, next_op, caller, gap in gaps:
            by_gap[(previous, next_op, caller)][0] += gap
            by_gap[(previous, next_op, caller)][1] += 1

        lines = [
            f"[glm52 host profile] iter={iteration} chunk={chunk} calls={len(self.calls)} "
            f"forward={forward_ns / 1e6:.3f} ms, TTNN calls={op_ns / 1e6:.3f} ms, "
            f"between calls={gap_ns / 1e6:.3f} ms, device wait={self.sync_ns / 1e6:.3f} ms",
            "[glm52 host profile] TTNN calls ranked by aggregate host time (ms, count, mean us, op):",
        ]
        for name, (total, count) in ranked(by_op.items()):
            lines.append(f"  {total / 1e6:8.3f} {count:5d} {total / count / 1e3:8.1f}  {name}")
        lines.append(
            "[glm52 host profile] Python gaps ranked by aggregate time (ms, count, mean us, previous -> next @ caller):"
        )
        for (previous, next_op, caller), (total, count) in ranked(by_gap.items()):
            lines.append(
                f"  {total / 1e6:8.3f} {count:5d} {total / count / 1e3:8.1f}  {previous} -> {next_op} @ {caller}"
            )
        lines.append("[glm52 host profile] Largest individual TTNN calls (ms, op, caller):")
        for name, caller, start, end in sorted(self.calls, key=lambda row: row[3] - row[2], reverse=True)[:20]:
            lines.append(f"  {(end - start) / 1e6:8.3f}  {name} @ {caller}")
        lines.append("[glm52 host profile] Largest individual Python gaps (ms, previous -> next @ caller):")
        for previous, next_op, caller, gap in sorted(gaps, key=lambda row: row[3], reverse=True)[:20]:
            lines.append(f"  {gap / 1e6:8.3f}  {previous} -> {next_op} @ {caller}")
        print("\n".join(lines), flush=True)
