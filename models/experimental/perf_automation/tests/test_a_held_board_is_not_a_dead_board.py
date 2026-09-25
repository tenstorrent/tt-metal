# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The run told the operator to reboot the host for a fault `tt-smi -r` cleared in ninety seconds.

VOXTRAL RUN 18, in order:

    [optimize/cc] reclaimed device (killed holders none) + no reset issued [DEVICE STILL UNHEALTHY]
    [optimize/cc] HALT -- reboot the host ... 'Failed to set initial power state'
                  -- REBOOT THE HOST, then re-run: RuntimeError: Read 0xffffffff over PCIe ID 2
    [optimize/supervisor] the attempt left 2 process(es) running after exiting (1047857, 1245943)
                  -- killing them before going on

Then, by hand: `tt-smi -r` -> all four p300c back at D0, ARC answering, 44-46 C. The board was never
unresettable. It was HELD. The resets that "failed" ran while something still had /dev/tenstorrent
open, and the reaping that would have freed it happened after the verdict was printed.

The kernel signature the halt quotes does not contradict this -- it supports it.
board_needs_host_reboot's own docstring says the message "fires whenever a device is OPENED while its
ARC is not ready", which is exactly what a stale holder produces. It is read as an EXPLANATION after
resets fail, never as a gate, precisely because it is transient: 714 in a day on a wedged board, 4 in
an hour on a healthy one.

WHY THIS IS NOT A LOOSENED LIMIT. RESET_FAILS once reached 34 against a limit of 3 -- "a limit
counted in one place and enforced in none is not a limit" -- and that must not return. So the retry
fires at most ONCE per run, and only when reaping actually killed something. A holder that existed is
new evidence about why the earlier resets failed; a retry against a changed world is a different
experiment. Reap nothing and nothing is retried, because repeating an unchanged experiment is what
the limit exists to prevent.
"""
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


@pytest.fixture()
def dr():
    # As a PACKAGE module. device_recovery uses relative imports, so loading it by path raises
    # ImportError -- and conftest notes it is imported under two names, so the package form is the
    # one the tool actually runs.
    import agent.device_recovery as m

    m.RESET_FAILS["n"] = m.RESET_FAIL_LIMIT  # exhausted, as at the halt
    m._POST_REAP_RETRY["n"] = 0  # the retry is once per RUN; each test is a fresh run
    return m


def test_a_reaped_holder_earns_one_more_reset(dr, monkeypatch):
    """THE RUN-18 CASE: holders existed, so the earlier failures have an explanation that a retry can
    test. The board comes back and the run continues instead of waiting for a human."""
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [1047857, 1245943])
    monkeypatch.setattr(dr, "recover", lambda *a, **k: True)
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is True


def test_reaping_nothing_retries_nothing(dr, monkeypatch):
    """Nothing changed, so there is nothing new to test -- and the device is not touched at all."""
    touched = []
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "recover", lambda *a, **k: touched.append(1) or True)
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is False
    assert touched == [], "reset the device when nothing had been freed"


def test_it_fires_at_most_once_per_run(dr, monkeypatch):
    """The guard against reopening the 34-resets-against-a-limit-of-3 hole."""
    calls = []
    monkeypatch.setattr(dr, "reap_device_holders", lambda: calls.append(1) or [999])
    monkeypatch.setattr(dr, "recover", lambda *a, **k: True)
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is True
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is False
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is False
    assert len(calls) == 1, "reaped more than once"


def test_a_failed_retry_still_halts(dr, monkeypatch):
    """A genuinely dead board must still stop the run -- this adds one attempt, not an escape."""
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [123])
    monkeypatch.setattr(dr, "recover", lambda *a, **k: False)
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is False


def test_the_reap_is_best_effort(dr, monkeypatch):
    """This runs when the board is already in trouble; a reclaim that raises must not turn a
    recoverable wedge into a dead run."""

    def _boom():
        raise RuntimeError("no fuser")

    monkeypatch.setattr(dr, "reap_device_holders", _boom)
    assert dr.retry_once_after_reaping("t", lambda tgt: True) is False


def test_the_halt_path_tries_before_it_verdicts(dr):
    """Ordering is the whole bug: reap, re-probe, THEN decide. Asserted on the source because the
    alternative is wedging a real board to reproduce it."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i_retry = src.index("retry_once_after_reaping")
    i_halt = src.index('"needs_host_reboot"')
    assert i_retry < i_halt, "the reboot verdict is reached before the post-reap retry"
