# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Unit tests for check_eth_status decoding. These need no hardware - they feed synthetic base-firmware
# samples through the decoders. Wormhole values come from a live N300 read.

import os
import sys

import pytest

metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_home = os.path.join(metal_home, "tools", "triage")
sys.path.insert(0, triage_home)

import eth_status_sampling
from check_eth_status import (
    BASE_FW_NOT_PROGRESSING,
    CORE_UNREADABLE,
    LINK_CHANGED,
    LINK_DOWN,
    LINK_DOWN_STALE,
    MAILBOX_NOT_SERVICED,
    PORT_NOT_TRAINED,
    PORT_STATUS_INVALID,
    EthFinding,
    decode_blackhole_core,
    decode_wormhole_core,
)
from eth_status_sampling import (
    ALL_ONES,
    BH_BOOT_RESULTS,
    BH_BOOT_RESULTS_WORDS,
    BH_HEARTBEAT,
    BH_LOCAL_INFO,
    BH_MAILBOX,
    BH_PCS_STATUS,
    BH_PORT_STATUS,
    BH_RETRAIN_COUNT,
    BH_RX_LINK_UP,
    WH_BOOT_RESULTS,
    WH_BOOT_RESULTS_WORDS,
    WH_HEARTBEAT,
    WH_LAUNCH_ERISC_APP_FLAG,
    WH_LINK_ERROR_STATUS,
    WH_LINK_STATUS,
    WH_LOCAL_ETH_ID,
    WH_PORT_DISABLE_MASK,
    WH_RETRAIN_COUNT,
    WH_RETRAIN_FORCE,
    WH_SHARED_HEARTBEAT,
    WH_TRAIN_STATUS,
    EthCoreSample,
    EthStatusSampling,
)

WH_CHANNEL = 8


def wh_words(updates: dict[int, int] | None = None) -> dict[int, int]:
    """An N300 channel-8 core with a trained link, as read live (plus metal firmware running on it)."""
    words = {WH_BOOT_RESULTS + 4 * index: 0 for index in range(WH_BOOT_RESULTS_WORDS)}
    words.update(
        {
            WH_HEARTBEAT: 0xABCD8128,
            WH_PORT_DISABLE_MASK: 0xFCFF,
            WH_TRAIN_STATUS: 1,
            WH_LINK_ERROR_STATUS: 0,
            WH_LAUNCH_ERISC_APP_FLAG: 1,
            WH_LINK_STATUS: 6,
            WH_SHARED_HEARTBEAT: 0xABCD8122,
            WH_LOCAL_ETH_ID: 8,
        }
    )
    return {**words, **(updates or {})}


def bh_words(updates: dict[int, int] | None = None) -> dict[int, int]:
    """A healthy Blackhole 2-erisc core with no router: CI's most common row."""
    words = {BH_BOOT_RESULTS + 4 * index: 0 for index in range(BH_BOOT_RESULTS_WORDS)}
    words.update(
        {
            BH_PORT_STATUS: 1,
            BH_HEARTBEAT: 0xABCD1234,
            BH_RETRAIN_COUNT: 0,
            BH_RX_LINK_UP: 1,
            BH_LOCAL_INFO: 6 << 16,
            BH_MAILBOX: 0xD0E50002,
            BH_PCS_STATUS: 1,
        }
    )
    return {**words, **(updates or {})}


def sample(words: dict[int, int], timestamp: float) -> EthCoreSample:
    return EthCoreSample(timestamp=timestamp, words=words)


def wh(first: dict[int, int] | None, second: dict[int, int], burst=None, halted=False, channel=WH_CHANNEL):
    return decode_wormhole_core(
        channel, None if first is None else sample(first, 0.0), sample(second, 3.0), burst or [], halted
    )


def bh(first: dict[int, int] | None, second: dict[int, int], burst=None, halted=False):
    return decode_blackhole_core(
        5, None if first is None else sample(first, 0.0), sample(second, 3.0), burst or [], halted
    )


def messages(findings: list[EthFinding]) -> list[tuple[bool, str]]:
    return [(finding.is_error, finding.message) for finding in findings]


# Wormhole


def test_wh_healthy_link_on_n300():
    row, findings = wh(wh_words(), wh_words({WH_HEARTBEAT: 0xABCD67C4, WH_SHARED_HEARTBEAT: 0xABCD67BC}))
    assert (row.port_status, row.link, row.retrain_count, row.heartbeat) == ("Trained", "Up", 0, "Moving (base FW)")
    assert (row.channel, row.physical_channel, row.rx_link_up, row.mailbox) == (8, 8, "Up", None)
    assert row.sample_gap == 3.0
    assert findings == []


def test_wh_reads_the_real_retrain_and_link_words():
    # The old script read 0x1EE8 and 0x1EE0, both reserved words in boot_results_t.
    words = wh_words({WH_RETRAIN_COUNT: 2, 0x1EE8: 5, 0x1EE0: 0})
    row, _ = wh(words, {**words, WH_HEARTBEAT: 0xABCD0001, WH_RETRAIN_COUNT: 3})
    assert (row.retrain_count, row.retrain_delta, row.link) == (3, 1, "Up")


def test_wh_router_that_never_yields_is_not_flagged():
    # The router zeroes 0x1C at main-loop entry and writes 0xDCBA to 0x1F80 while it loops.
    first = wh_words({WH_HEARTBEAT: 0, WH_SHARED_HEARTBEAT: 0xDCBA0040})
    second = wh_words({WH_HEARTBEAT: 0, WH_SHARED_HEARTBEAT: 0xDCBA0080})
    row, findings = wh(first, second)
    assert (row.heartbeat, row.link) == ("Moving (router)", "Up (stale)")
    assert findings == []


def test_wh_link_down_with_fresh_evidence_is_an_error():
    first = wh_words({WH_LINK_STATUS: 3})
    second = wh_words({WH_LINK_STATUS: 3, WH_HEARTBEAT: 0xABCD9000})
    row, findings = wh(first, second)
    assert (row.link, row.rx_link_up, row.link_raw) == ("Down", "Down", 3)
    assert messages(findings) == [(True, LINK_DOWN)]


def test_wh_link_down_in_a_stale_snapshot_is_a_warning():
    words = wh_words({WH_LINK_STATUS: 3})
    row, findings = wh(words, dict(words))
    assert (row.link, row.heartbeat) == ("Down (stale)", "Frozen")
    assert messages(findings) == [(False, LINK_DOWN_STALE)]


def test_wh_link_that_changed_during_sampling_is_a_warning():
    second = wh_words({WH_LINK_STATUS: 3, WH_HEARTBEAT: 0xABCD9000})
    row, findings = wh(wh_words(), second)
    assert row.link == "Up -> Down"
    assert messages(findings) == [(False, LINK_CHANGED)]


@pytest.mark.parametrize(
    "updates, port_status, finding",
    [
        ({WH_PORT_DISABLE_MASK: 0xFFFF}, "Disabled", PORT_NOT_TRAINED),
        ({WH_RETRAIN_FORCE: 1}, "Retraining", PORT_NOT_TRAINED),
        ({WH_TRAIN_STATUS: 0}, "Training", PORT_NOT_TRAINED),
        ({WH_TRAIN_STATUS: 2, WH_LINK_ERROR_STATUS: 12}, "Not connected", PORT_NOT_TRAINED),
        ({WH_TRAIN_STATUS: 2, WH_LINK_ERROR_STATUS: 5}, "Train failed", PORT_NOT_TRAINED),
        ({WH_TRAIN_STATUS: 9}, "Invalid", PORT_STATUS_INVALID),
        ({WH_TRAIN_STATUS: ALL_ONES}, "Unreadable", None),
    ],
)
def test_wh_port_status(updates, port_status, finding):
    second = wh_words({**updates, WH_HEARTBEAT: 0xABCD9000})
    row, findings = wh(wh_words(updates), second)
    assert row.port_status == port_status
    assert messages(findings) == ([] if finding is None else [(False, finding)])


def test_wh_frozen_base_firmware_without_metal_firmware_is_a_warning():
    words = wh_words({WH_LAUNCH_ERISC_APP_FLAG: 0})
    frozen_burst = [{WH_HEARTBEAT: 0xABCD8128, WH_SHARED_HEARTBEAT: 0xABCD8122}] * 8
    row, findings = wh(words, dict(words), burst=frozen_burst)
    assert row.heartbeat == "Frozen"
    assert (False, BASE_FW_NOT_PROGRESSING) in messages(findings)


def test_wh_burst_catches_a_heartbeat_that_matched_by_chance():
    words = wh_words({WH_LAUNCH_ERISC_APP_FLAG: 0})
    row, findings = wh(words, dict(words), burst=[{WH_HEARTBEAT: 0xABCD8129, WH_SHARED_HEARTBEAT: 0xABCD8123}])
    assert (row.heartbeat, row.link) == ("Moving (base FW)", "Up")
    assert findings == []


def test_wh_core_halted_by_triage_is_not_called_stuck():
    words = wh_words({WH_LAUNCH_ERISC_APP_FLAG: 0})
    row, findings = wh(words, dict(words), halted=True)
    assert row.heartbeat == "Frozen (halted by triage)"
    assert findings == []


def test_wh_unreadable_core_is_a_warning_not_an_error():
    words = {address: ALL_ONES for address in wh_words()}
    row, findings = wh(words, dict(words))
    assert (row.port_status, row.link, row.heartbeat, row.retrain_count) == (
        "Unreadable",
        "Unreadable",
        "Unreadable",
        None,
    )
    assert messages(findings) == [(False, CORE_UNREADABLE)]


def test_wh_single_sample_raises_no_link_finding():
    row, findings = wh(None, wh_words({WH_LINK_STATUS: 3}))
    assert (row.heartbeat, row.link, row.sample_gap) == ("One sample", "Down", None)
    assert findings == []


# Blackhole


def test_bh_healthy_two_erisc_core_without_a_router():
    row, findings = bh(bh_words(), bh_words())
    assert (row.port_status, row.link, row.heartbeat, row.mailbox) == ("Up", "Up", "Frozen", "HOST: DONE RELEASE_CORE")
    assert (row.channel, row.physical_channel, row.rx_link_up, row.link_raw) == (5, 6, "Up", 1)
    assert findings == []


def test_bh_retrain_count_is_reported_but_never_a_finding():
    # CI raised 38 ERRORs for 'retrain count is 1' on links that were up.
    row, findings = bh(bh_words({BH_RETRAIN_COUNT: 1}), bh_words({BH_RETRAIN_COUNT: 1}))
    assert (row.retrain_count, row.retrain_delta) == (1, 0)
    assert findings == []


def test_bh_live_link_down_is_an_error():
    row, findings = bh(bh_words({BH_PCS_STATUS: 0}), bh_words({BH_PCS_STATUS: 0}))
    assert (row.link, row.port_status) == ("Down", "Up")
    assert messages(findings) == [(True, LINK_DOWN)]


def test_bh_unused_port_link_down_is_not_an_error():
    words = bh_words({BH_PORT_STATUS: 3, BH_PCS_STATUS: 0})
    row, findings = bh(words, dict(words))
    assert (row.port_status, row.link) == ("Unused", "Down")
    assert findings == []


def test_bh_falls_back_to_the_snapshot_when_pcs_is_unreadable():
    words = bh_words({BH_PCS_STATUS: ALL_ONES, BH_RX_LINK_UP: 0})
    row, findings = bh(words, dict(words))
    assert (row.link, row.rx_link_up) == ("Down (stale)", "Down")
    assert messages(findings) == [(False, LINK_DOWN_STALE)]


def test_bh_all_ones_rx_link_up_is_unreadable_not_up():
    words = bh_words({BH_PCS_STATUS: ALL_ONES, BH_RX_LINK_UP: ALL_ONES})
    row, findings = bh(words, dict(words))
    assert (row.link, row.rx_link_up) == ("Unreadable", "Unreadable")
    assert findings == []


@pytest.mark.parametrize(
    "value, port_status, finding",
    [(0, "Unknown", PORT_NOT_TRAINED), (7, "Invalid", PORT_STATUS_INVALID), (ALL_ONES, "Unreadable", None)],
)
def test_bh_port_status(value, port_status, finding):
    # Unlike the old script, an unknown or invalid port no longer hides the rest of the row.
    words = bh_words({BH_PORT_STATUS: value})
    row, findings = bh(words, dict(words))
    assert (row.port_status, row.link, row.retrain_count) == (port_status, "Up", 0)
    assert messages(findings) == ([] if finding is None else [(False, finding)])


def test_bh_mailbox_slots_are_16_bytes_apart():
    # The old script read 0x7D004..0x7D00C, which are the HOST slot's arguments, not other slots.
    words = bh_words({BH_MAILBOX + 4: 0xCA110009, BH_MAILBOX + 0x10: 0xCA11000F})
    row, _ = bh(bh_words(), words)
    assert row.mailbox == "HOST: DONE RELEASE_CORE, RISC1: CALL DYNAMIC_NOC_INIT"
    assert row.mailbox_raw is not None and row.mailbox_raw.split()[4] == "0xca11000f"


def test_bh_mailbox_request_pending_in_both_samples_warns_once():
    words = bh_words({BH_MAILBOX: 0xCA110009, BH_MAILBOX + 0x20: 0xCEDE0001})
    row, findings = bh(words, dict(words))
    assert row.mailbox == "HOST: CALL PORT_ACTION, CMFW: CALL_ACK LINK_STATUS_CHECK"
    assert messages(findings) == [(False, MAILBOX_NOT_SERVICED)]


def test_bh_mailbox_request_serviced_during_sampling_is_fine():
    _, findings = bh(bh_words({BH_MAILBOX: 0xCA110009}), bh_words({BH_MAILBOX: 0xD0E50009}))
    assert findings == []


@pytest.mark.parametrize(
    "before, after, heartbeat",
    [
        (0xDCBA0040, 0xDCBA0080, "Moving (router)"),
        (0xABCD0001, 0xABCD0002, "Moving (base FW)"),
        (0xABCD0001, 0xDCBA0040, "Moving (router)"),
        (0x00000000, 0x12345678, "Moving"),
    ],
)
def test_bh_heartbeat_writers(before, after, heartbeat):
    row, _ = bh(bh_words({BH_HEARTBEAT: before}), bh_words({BH_HEARTBEAT: after}))
    assert row.heartbeat == heartbeat


def test_bh_unreadable_core_is_a_warning_not_an_error():
    words = {address: ALL_ONES for address in bh_words()}
    row, findings = bh(bh_words(), words)
    assert (row.port_status, row.link, row.heartbeat) == ("Unreadable", "Unreadable", "Unreadable")
    assert messages(findings) == [(False, CORE_UNREADABLE)]


# Sampling


def test_second_sample_waits_for_the_minimum_gap(monkeypatch):
    sampling = object.__new__(EthStatusSampling)
    sampling.initial_samples = {"core": sample({}, timestamp=10.0)}
    slept: list[float] = []
    monkeypatch.setattr(eth_status_sampling.time, "monotonic", lambda: 11.0)
    monkeypatch.setattr(eth_status_sampling.time, "sleep", slept.append)
    sampling.wait_for_second_sample()
    assert slept == [pytest.approx(eth_status_sampling.MINIMUM_SAMPLE_GAP_SECONDS - 1.0)]
