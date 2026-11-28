"""A crashed PCC test hands the recovery policy the line that says the board is dead, not a warnings tail.

Qwen-Image-Edit 2026-10-02: the e2e test died at mesh open with "Timed out waiting for ETH heartbeat".
check_pcc handed the policy the last 2000 characters of pytest's output -- its warnings summary, printed
AFTER the traceback -- so is_dead_board saw no signature, the telemetry veto skipped the reset, and
every later round met the same stuck fabric. The evidence lines now lead the excerpt.
"""

from __future__ import annotations

from agent import device_recovery as DR
from agent import pcc_runner as PR

_FAULT = (
    "E       RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 7, ETH core e8-6 (NOC0) "
    "to advance. Stuck at 0xaabb0024"
)
_WARNINGS = "\n".join(
    "  /venv/lib/python3.10/site-packages/pkg/mod_%d.py:62: DeprecationWarning: this will go away" % i
    for i in range(80)
)


def _pytest_output(fault=_FAULT):
    return "\n".join(
        [
            "==================================== ERRORS ====================================",
            "____________ ERROR at setup of test_e2e[mesh_device0-device_params0] ____________",
            fault,
            "=============================== warnings summary ===============================",
            _WARNINGS,
            "=========================== short test summary info ============================",
            "ERROR test_e2e.py::test_e2e[mesh_device0-device_params0] - RuntimeError: Timed out wai",
        ]
    )


def test_the_old_tail_lost_the_evidence():
    """The premise, kept checkable: the plain 2000-char tail the policy used to get is not a dead board."""
    out = _pytest_output()
    assert DR.is_dead_board(out)
    assert not DR.is_dead_board(out.strip()[-2000:])


def test_the_excerpt_now_leads_with_it_and_the_policy_sees_a_dead_board():
    excerpt = PR._useful_tail(_pytest_output())
    head = excerpt.split("warnings summary")[0]
    assert "ERROR at setup of" in head and "Timed out waiting for ETH heartbeat" in head
    assert DR.is_dead_board(excerpt)
    assert excerpt.rstrip().endswith("RuntimeError: Timed out wai"), "the tail is still there, after it"


def test_an_ordinary_failure_keeps_exactly_the_old_excerpt():
    out = "\n".join(["E   AssertionError: Gate 3: min image PCC 0.94 < 0.95", _WARNINGS])
    kept = "\n".join(ln for ln in out.splitlines() if not PR._TEARDOWN_NOISE.search(ln)).strip()[-2000:]
    assert PR._useful_tail(out) == kept


def test_a_signature_already_in_the_tail_is_not_repeated():
    out = "noise\n" + _FAULT
    assert PR._useful_tail(out).count("ETH heartbeat") == 1


def test_the_evidence_is_the_same_verdict_as_the_whole_text():
    samples = [
        _pytest_output(),
        _pytest_output(fault="E   RuntimeError: TT_FATAL @ mesh_device.cpp:1 open failed"),  # bring-up rule
        "Read 0xffffffff over PCIe ID 3\n" + _WARNINGS,
        "E   AssertionError: pcc 0.9\n" + _WARNINGS,
        "",
    ]
    for t in samples:
        assert DR.is_dead_board(DR.dead_board_evidence(t)) == DR.is_dead_board(t), t[:80]


def test_the_evidence_is_bounded_and_deduplicated():
    t = "\n".join([_FAULT] * 50 + ["E RuntimeError: x%d" % i for i in range(50)])
    ev = DR.dead_board_evidence(t, limit=5).splitlines()
    assert len(ev) == 5 and ev.count(_FAULT.strip()) == 1
