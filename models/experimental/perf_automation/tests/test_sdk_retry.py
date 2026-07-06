import pytest

from agent.sdk_retry import run_with_retry


def test_transient_then_success():
    n = {"c": 0}

    async def go():
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("Command failed with exit code 129")

    run_with_retry(go, attempts=5, base_sleep=0)
    assert n["c"] == 3


def test_non_transient_raises_immediately():
    n = {"c": 0}

    async def go():
        n["c"] += 1
        raise ValueError("bad json")

    with pytest.raises(ValueError):
        run_with_retry(go, attempts=5, base_sleep=0)
    assert n["c"] == 1


def test_exhausts_attempts_then_raises():
    n = {"c": 0}

    async def go():
        n["c"] += 1
        raise RuntimeError("message reader exit code 129")

    with pytest.raises(RuntimeError):
        run_with_retry(go, attempts=2, base_sleep=0)
    assert n["c"] == 2


def test_reset_runs_each_attempt():
    r = {"c": 0}
    n = {"c": 0}

    async def go():
        n["c"] += 1
        if n["c"] < 2:
            raise RuntimeError("exit code 129")

    run_with_retry(go, reset=lambda: r.__setitem__("c", r["c"] + 1), attempts=3, base_sleep=0)
    assert r["c"] == 2
