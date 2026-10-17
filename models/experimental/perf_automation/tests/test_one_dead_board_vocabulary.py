# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: every path that resets a board recognises a dead board the SAME way.

device_recovery.DEAD_BOARD_SIGS is the tool's list of "the card stopped answering" signatures.
perf_test_gen kept a SECOND, private list (_DEVICE_DISRUPTION_RE) for the same question, and the two
drifted: the ETH-fabric wedge was added to the first and was never on the second.

The reset in _run_perf_node's retry loop already existed and worked. It was simply never reached.
Measured 2026-09-25, the perf-test builder against Qwen-Image-Edit on a T3K:

    E  RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID 10226921817,
       ETH core e7-0 (NOC0) to advance. Stuck at 0xaabb0024
    E  Location: .../umd/device/pcie/pci_device.cpp:442

three times in a row at device open, no reset between them, 51 minutes spent. Measured directly:

    device_recovery.is_dead_board(text)        -> True
    perf_test_gen._is_device_disruption(1, t)  -> False

The builder's own agent wrote the diagnosis into its log: "The harness only resets the board after a
hang, not on this failure, and no other process is holding the chips."

What is pinned here: the predicate ASKS the shared one (so a signature added anywhere reaches every
caller), and the narrowing guards that make it safe are unchanged.
"""

from __future__ import annotations

import inspect

import pytest

from models.experimental.perf_automation.agent import device_recovery
from models.experimental.perf_automation.agent import perf_test_gen

# Verbatim from the run's log, including the UMD location line that follows it.
ETH_WEDGE_AT_OPEN = (
    "E       RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID 10226921817, "
    "ETH core e7-0 (NOC0) to advance. Stuck at 0xaabb0024\n"
    "E       Location: /home/ttuser/apande/tt-metal/tt_metal/third_party/umd/device/pcie/"
    "pci_device.cpp:442\n"
)


def test_the_wedge_that_cost_fifty_one_minutes_is_now_a_disruption():
    """The exact text, and the exact call, that returned False for a whole discovery step."""
    assert device_recovery.is_dead_board(ETH_WEDGE_AT_OPEN), "the shared list stopped recognising it"
    assert perf_test_gen._is_device_disruption(1, ETH_WEDGE_AT_OPEN) is True


def test_both_predicates_now_agree_on_every_shared_signature():
    """The point of the change: one vocabulary. Anything device_recovery calls dead, and that did
    NOT already run the test body, must reach the reset in the builder's retry loop too."""
    for sig in device_recovery.DEAD_BOARD_SIGS:
        text = f"E       RuntimeError: device failure at open -- {sig} -- aborting\n"
        assert device_recovery.is_dead_board(text), sig
        assert perf_test_gen._is_device_disruption(1, text) is True, sig


def test_the_signatures_are_not_copied_into_the_second_file():
    """A second list is the bug. Reuse the predicate; do not re-spell what it matches."""
    src = inspect.getsource(perf_test_gen._is_device_disruption)
    assert "is_dead_board(out)" in src
    for sig in device_recovery.DEAD_BOARD_SIGS:
        assert sig not in src, f"{sig!r} was copied in instead of being asked for"
    whole = inspect.getsource(perf_test_gen)
    assert "eth heartbeat" not in whole.lower().replace(
        "eth heartbeat on device asic id", ""
    ), "the signature leaked into this module's own matching"


# ------------------------------------------------------------------ the guards must still hold


@pytest.mark.parametrize("marker", perf_test_gen._TRACE_RAN_MARKERS)
def test_a_trace_hang_is_still_not_retried_here(marker):
    """THE REGRESSION THIS CHANGE COULD HAVE CAUSED, pinned.

    A test that already RAN has had its one reset; it must come back as a WEDGE, not loop
    reset+retry, "which just re-hangs". A hang's output can perfectly well also name a dead board,
    so the new check sits BELOW that guard -- asking first would have defeated it."""
    text = ETH_WEDGE_AT_OPEN + f"\n{marker} something\n"
    assert device_recovery.is_dead_board(text), "precondition: the text does name a dead board"
    assert (
        perf_test_gen._is_device_disruption(124, text) is False
    ), "a trace hang was reclassified as a fresh disruption and will now loop reset+retry"


def test_an_ordinary_test_bug_still_flows_to_the_correction_loop():
    """Kept narrow: an assertion or import error is the AGENT's to fix, not a reason to reset."""
    for text in (
        "E       AssertionError: expected 4 outputs, got 3",
        "E       ImportError: cannot import name 'prompt_ids_for_isl'",
        "E       ModuleNotFoundError: No module named 'models.tt_dit.pipelines.qwen_image_edit_vae'",
        "1 failed, 0 passed in 12.00s",
    ):
        assert perf_test_gen._is_device_disruption(1, text) is False, text


def test_empty_output_is_not_a_disruption():
    for text in ("", None):
        assert perf_test_gen._is_device_disruption(1, text) is False, repr(text)


def test_the_signatures_it_already_matched_still_match():
    """The private regex stays -- it holds board faults that are NOT 'the card stopped answering'
    (a clock that would not settle, a sysmem mapping). Nothing here removes them."""
    for text in (
        "AICLK failed to settle after reset",
        "clamped by max-arbiter",
        "Sysmem mapped at unexpected NOC address",
        "failed to open device 3",
        "GetPCIeDeviceID returned an error",
    ):
        assert perf_test_gen._is_device_disruption(1, text) is True, text
