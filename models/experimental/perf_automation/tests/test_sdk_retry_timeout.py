# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""run_with_retry must BOUND a stalled agentic call (no exception, just blocks) instead of
freezing forever — the PLAN-hang root cause. Timeout -> transient -> retry -> finally raise."""

import asyncio
import time

import pytest

from agent.sdk_retry import is_transient, run_with_retry


def test_normal_call_completes_once():
    calls = {"n": 0}

    async def go():
        calls["n"] += 1
        await asyncio.sleep(0.01)

    run_with_retry(go, attempts=3, timeout=2.0)
    assert calls["n"] == 1


def test_permanent_hang_is_bounded_not_frozen():
    seen = {"n": 0}

    async def go():
        seen["n"] += 1
        await asyncio.sleep(9999)  # stalled round

    t0 = time.time()
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        run_with_retry(go, attempts=3, base_sleep=0.0, timeout=0.3)
    dt = time.time() - t0
    assert seen["n"] == 3, seen  # retried, not retried-forever
    assert dt < 3.0, dt  # bounded (~0.9s), NOT a freeze


def test_one_shot_hang_recovers_on_retry():
    state = {"n": 0}

    async def go():
        state["n"] += 1
        if state["n"] == 1:
            await asyncio.sleep(9999)
        await asyncio.sleep(0.01)

    run_with_retry(go, attempts=3, base_sleep=0.0, timeout=0.3)
    assert state["n"] == 2


def test_nontransient_error_raises_without_wasteful_retries():
    state = {"n": 0}

    async def go():
        state["n"] += 1
        raise ValueError("model files not found")

    with pytest.raises(ValueError):
        run_with_retry(go, attempts=3, timeout=2.0)
    assert state["n"] == 1


def test_timeout_is_classified_transient():
    assert is_transient(asyncio.TimeoutError()) is True
    assert is_transient(TimeoutError()) is True
    assert is_transient(Exception("Command failed with exit code 129")) is True
    assert is_transient(ValueError("model files not found")) is False
