# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from models.tt_dit.utils.watchdog import run_with_stall_watchdog


def test_returns_value_without_stall():
    assert run_with_stall_watchdog(lambda: 7, name="t", stall_s=5, poll_s=0.01) == 7


def test_retries_an_exception_once():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "ok"

    assert run_with_stall_watchdog(fn, name="t", stall_s=0) == "ok"
    assert len(calls) == 2


def test_second_exception_propagates(expect_error):
    def fn():
        raise RuntimeError("persistent")

    with expect_error(RuntimeError, "persistent"):
        run_with_stall_watchdog(fn, name="t", stall_s=0)


def test_stall_without_progress_fires():
    fired = threading.Event()
    run_with_stall_watchdog(
        lambda: fired.wait(2),
        name="t",
        stall_s=0.05,
        poll_s=0.01,
        progress=lambda: 0.0,
        on_stall=lambda name, s: fired.set(),
    )
    assert fired.is_set()


def test_progress_keeps_a_slow_step_alive():
    fired = threading.Event()
    ticks = iter(range(10**6))
    run_with_stall_watchdog(
        lambda: time.sleep(0.3),
        name="t",
        stall_s=0.05,
        poll_s=0.01,
        progress=lambda: float(next(ticks)),
        on_stall=lambda name, s: fired.set(),
    )
    assert not fired.is_set()
