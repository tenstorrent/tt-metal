# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: a board that cannot be opened is dead whatever the runtime called it.

DEAD_BOARD_SIGS is an allowlist of PROSE. Four of its entries were each added by hand AFTER a run
had already been lost to the wording they match, and the list was still one short every time:

    "Timed out waiting for ETH heartbeat ... Stuck at 0xaabb0024"   added by hand
    "Firmware startup error on device N at core X over NOC0"        added by hand
    "NOC0 is hung on PCIe device ID 9."                             added by hand
    "Timeout waiting for Ethernet core service remote IO request."  NOT matched

The last one, measured 2026-09-25 on a T3K: the profiler died on it 18 times in a row (the agent
transcript records 18x "profiler crashed: no ops_perf_results_*.csv produced"), is_dead_board
returned False every time, no reset was ever issued, and the run spent its entire budget with
"nothing recorded yet".

An allowlist cannot be completed by adding to it. So the question asked is WHERE the failure
happened, not what it was called: pytest reports a fixture failure as "ERROR at setup of <node>",
and the only thing setup does for these tests is bring the device up.

The excerpt below is verbatim from the log that killed that run
(captured before the harness deleted its temp dir).
"""

from __future__ import annotations

import inspect

import pytest

from models.experimental.perf_automation.agent import device_recovery as dr

# Verbatim, from /tmp/perf_mcp_*/run0_tracy.log of the 11:33 run on 2026-09-25.
REAL_CRASH = """==================================== ERRORS ====================================
__ ERROR at setup of test_main_perf[wormhole_b0-mesh_device0-device_params0] ___

    @pytest.fixture(scope="function")
    def mesh_device(request, silicon_arch_name, device_params):
E       RuntimeError: Timeout waiting for Ethernet core service remote IO request.

/home/ttuser/apande/tt-metal/ttnn/ttnn/distributed/distributed.py:631: RuntimeError
"""


def test_the_wording_that_cost_a_whole_run_is_recognised():
    """The failure this change exists for. It matches no signature, and it must still be a wedge."""
    assert not any(
        sig in REAL_CRASH.lower() for sig in dr.DEAD_BOARD_SIGS
    ), "precondition: this text must NOT match the allowlist, or the test proves nothing"
    assert dr.is_dead_board(REAL_CRASH) is True


def test_a_wording_nobody_has_seen_yet_is_recognised_too():
    """The actual point: the next wording does not need a code change."""
    future = (
        "==================================== ERRORS =========================\n"
        "__ ERROR at setup of test_main_perf[wormhole_b0-mesh_device0] ___\n"
        "E       RuntimeError: some fabric fault nobody has written down yet\n"
    )
    assert not any(sig in future.lower() for sig in dr.DEAD_BOARD_SIGS)
    assert dr.is_dead_board(future) is True


def test_every_signature_that_matched_before_still_matches():
    """Additive only: the allowlist answers first and is unchanged."""
    for sig in dr.DEAD_BOARD_SIGS:
        assert dr.is_dead_board(f"RuntimeError: device failure -- {sig} -- aborting"), sig


# ------------------------------------------------------------------ deliberately narrow


@pytest.mark.parametrize(
    "exc",
    [
        "ImportError: cannot import name 'prompt_ids_for_isl'",
        "ModuleNotFoundError: No module named 'models.tt_dit.pipelines.qwen_image_edit_vae'",
        "FileNotFoundError: [Errno 2] No such file or directory: 'weights.bin'",
        "AttributeError: 'NoneType' object has no attribute 'shape'",
    ],
)
def test_a_host_problem_at_setup_is_not_a_dead_board(exc):
    """Setup failing is not enough -- it must be a HARD runtime fault. A missing module or file is
    the operator's problem and resetting the board would hide it."""
    text = f"__ ERROR at setup of test_main_perf[wormhole_b0-mesh_device0] ___\nE       {exc}\n"
    assert dr.is_dead_board(text) is False, exc


def test_a_failure_inside_the_test_body_is_never_a_dead_board():
    """No setup report => the device opened fine => whatever failed is not the board."""
    for text in (
        "E       RuntimeError: Timeout waiting for Ethernet core service remote IO request.",
        "no TRACE_PER_TOKEN_MS or FORWARD_WALL_MS in output (workload did not run full-pipeline)",
        "Gate 3: min image PCC 0.720134 < 0.99",
        "E       AssertionError: expected 4 outputs, got 3",
        "1 failed, 0 passed in 12.00s",
        "",
        None,
    ):
        assert dr.is_dead_board(text) is False, repr(text)


def test_both_halves_are_required():
    """The report line alone, or the fault alone, is not enough."""
    assert dr.is_device_bringup_failure("__ ERROR at setup of test_x ___\nsomething went wrong") is False
    assert dr.is_device_bringup_failure("E  RuntimeError: boom") is False
    assert dr.is_device_bringup_failure("__ ERROR at setup of test_x ___\nE  RuntimeError: boom") is True


def _code_of(fn) -> str:
    """A function's source with its docstring removed -- prose about the change must not satisfy
    (or break) an assertion about the CODE."""
    src = inspect.getsource(fn)
    doc = inspect.getdoc(fn)
    if doc:
        for line in doc.splitlines():
            src = src.replace(line, "")
    return src


def test_the_structural_check_does_not_re_spell_the_allowlist():
    """No fifth list: the new tier must add a QUESTION, not more prose to match."""
    body = _code_of(dr.is_device_bringup_failure).lower()
    for sig in dr.DEAD_BOARD_SIGS:
        assert sig not in body, f"{sig!r} was copied into the structural check"
    assert "setup" in body, "the check must key off WHERE the failure happened"


def test_the_allowlist_still_answers_first():
    """Order matters for cost: a recognised signature must not depend on the structural tier."""
    code = _code_of(dr.is_dead_board)
    i_sig = code.index("DEAD_BOARD_SIGS")
    i_struct = code.index("is_device_bringup_failure(")
    assert i_sig < i_struct, "the cheap allowlist check must run before the structural one"


def test_a_trace_hang_is_still_not_reclassified():
    """The perf-test builder's guard must still win: a hang that already ran the body has had its
    one reset and must return as a WEDGE, not loop reset+retry."""
    from models.experimental.perf_automation.agent import perf_test_gen

    hung = REAL_CRASH + "\nFORWARD_WALL_MS=1234\n"
    assert perf_test_gen._is_device_disruption(124, hung) is False
