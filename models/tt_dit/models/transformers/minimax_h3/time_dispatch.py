# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""``MINIMAX_H3_TIME_DISPATCH=<block>`` logs the host time of every top-level ttnn call of that block per
forward (enqueue cost only, never device time)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import ExitStack

from loguru import logger

import ttnn

_ENV = "MINIMAX_H3_TIME_DISPATCH"


def timed_block_index() -> int | None:
    """The block index to time, or None when the timer is off."""
    raw = os.environ.get(_ENV, "0")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{_ENV}={raw!r} must be an integer block index") from None
    return None if raw == "0" or value < 0 else (0 if raw == "1" else value)


class DispatchTimer:
    """Times the top-level ttnn ops dispatched inside the ``with`` body and logs a per-kind table."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._stack = ExitStack()
        self._ops: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # name -> [calls, ns]
        self._syncs = 0
        self._gap_ns = 0
        self._t_pre = 0
        self._t_last_post: int | None = None
        self._sync = None

    def __enter__(self) -> "DispatchTimer":
        self._sync = ttnn.synchronize_device

        def synchronize_device(*args, **kwargs):
            self._syncs += 1
            t0 = time.perf_counter_ns()
            result = self._sync(*args, **kwargs)
            self._ops["synchronize_device"][0] += 1
            self._ops["synchronize_device"][1] += time.perf_counter_ns() - t0
            return result

        ttnn.synchronize_device = synchronize_device
        self._stack.enter_context(ttnn.decorators.register_pre_operation_hook(self._pre))
        self._stack.enter_context(ttnn.decorators.register_post_operation_hook(self._post))
        self._t_start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()
        ttnn.synchronize_device = self._sync
        if exc[0] is None:
            self._report(time.perf_counter_ns() - self._t_start)

    def _pre(self, operation, args, kwargs) -> None:
        self._t_pre = time.perf_counter_ns()
        if self._t_last_post is not None:
            self._gap_ns += self._t_pre - self._t_last_post

    def _post(self, operation, args, kwargs, output) -> None:
        now = time.perf_counter_ns()
        entry = self._ops[operation.python_fully_qualified_name]
        entry[0] += 1
        entry[1] += now - self._t_pre
        self._t_last_post = now

    def _report(self, total_ns: int) -> None:
        if ttnn.CONFIG.enable_fast_runtime_mode:
            # FastOperation skips the hooks: only the wall time of the block is known.
            logger.info(
                f"dispatch timing [{self.label}]: {total_ns / 1e3:.0f} us wall, {self._syncs} synchronize_device; "
                "the per-op table needs TTNN_CONFIG_OVERRIDES='{\"enable_fast_runtime_mode\": false}'"
            )
            return
        rows = sorted(self._ops.items(), key=lambda kv: kv[1][1], reverse=True)
        op_ns = sum(ns for _, ns in self._ops.values())
        lines = [
            f"dispatch timing [{self.label}]: {sum(c for c, _ in self._ops.values())} ttnn calls, "
            f"{op_ns / 1e3:.0f} us in ops, {self._gap_ns / 1e3:.0f} us in Python gaps, "
            f"{total_ns / 1e3:.0f} us wall, {self._syncs} synchronize_device",
            f"  {'op':<58}{'calls':>6}{'us total':>10}{'us/call':>9}",
        ]
        for name, (calls, ns) in rows:
            lines.append(f"  {name:<58}{calls:>6}{ns / 1e3:>10.0f}{ns / 1e3 / calls:>9.1f}")
        logger.info("\n".join(lines))
