#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_eth_status.py

Description:
    Checks the link on every active ethernet core (the cores metal found trained when it started).
    Base-firmware state is read twice, at least 3 s apart; eth_status_sampling takes the first sample early.
    Reports port and link state, retrain count, heartbeat progress and, on Blackhole, the eth firmware mailboxes.
    A link that is down in both samples on live or freshly refreshed evidence is an error. Retrain counts are
    reported but never flagged.

Owner:
    nhuang-tt
"""

from dataclasses import dataclass

from eth_status_sampling import (
    ALL_ONES,
    BH_BOOT_RESULTS,
    BH_BOOT_RESULTS_WORDS,
    BH_HEARTBEAT,
    BH_LOCAL_INFO,
    BH_MAILBOX,
    BH_MAILBOX_SLOT_SIZE,
    BH_MAILBOX_SLOTS,
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
    WH_LINK_UP,
    WH_LOCAL_ETH_ID,
    WH_PORT_DISABLE_MASK,
    WH_RETRAIN_COUNT,
    WH_RETRAIN_FORCE,
    WH_SHARED_HEARTBEAT,
    WH_TRAIN_STATUS,
    EthCoreSample,
    EthStatusSampling,
    get_read_plan,
    run as get_eth_status_sampling,
)
from run_checks import run as get_run_checks
from triage import (
    ScriptConfig,
    hex_serializer,
    log_check_location,
    log_warning_location,
    run_script,
    triage_field,
)
from triage_session import get_triage_session
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
import utils

script_config = ScriptConfig(
    depends=["run_checks", "eth_status_sampling"],
)

# Top 16 bits of a heartbeat word name its writer.
BASE_FW_SIGNATURE = 0xABCD
ROUTER_SIGNATURE = 0xDCBA
WRITER_NAMES = ((BASE_FW_SIGNATURE, "base FW"), (ROUTER_SIGNATURE, "router"))

WH_TRAIN_STATUS_NAMES = {0: "Training", 1: "Trained", 2: "Train failed"}
# Link error codes from here up mean nothing is plugged in (UMD ETH_LINK_UNUSED_ERROR_CODE_RANGE_START).
WH_NOT_CONNECTED_ERROR_CODE = 11
WH_NOT_TRAINED_STATES = ("Disabled", "Retraining", "Training", "Train failed", "Not connected")
BH_PORT_STATUS_NAMES = {0: "Unknown", 1: "Up", 2: "Down", 3: "Unused"}

MAILBOX_SLOT_NAMES = ("HOST", "RISC1", "CMFW", "OTHER")
MAILBOX_CALL = 0xCA11  # posted, not picked up yet
MAILBOX_CALL_ACK = 0xCEDE  # picked up, handler still running (eth FW >= 1.5.0)
MAILBOX_STATUS_NAMES = {MAILBOX_CALL: "CALL", MAILBOX_CALL_ACK: "CALL_ACK", 0xD0E5: "DONE"}
MAILBOX_TYPE_NAMES = {
    0x1: "LINK_STATUS_CHECK",
    0x2: "RELEASE_CORE",
    0x6: "PORT_REINIT_MACPCS",
    0x9: "PORT_ACTION",
    0xF: "DYNAMIC_NOC_INIT",
}

# Finding messages carry no values, so sqlite can group by message; values stay in the columns.
LINK_DOWN = "Eth link is down"
LINK_DOWN_STALE = "Eth link reported down by a stale snapshot"
LINK_CHANGED = "Eth link state changed during sampling"
PORT_NOT_TRAINED = "Eth port is not trained"
PORT_STATUS_INVALID = "Eth port status is invalid"
MAILBOX_NOT_SERVICED = "Eth firmware mailbox request not serviced"
BASE_FW_NOT_PROGRESSING = "Eth base firmware is not progressing"
CORE_UNREADABLE = "Eth core is unreadable"


@dataclass
class EthCoreCheckData:
    channel: int | None = triage_field("Chan")
    port_status: str | None = triage_field("Port Status")
    link: str | None = triage_field("Link")
    retrain_count: int | None = triage_field("Retrain Count")
    heartbeat: str | None = triage_field("Heartbeat")
    mailbox: str | None = triage_field("Mailbox")
    physical_channel: int | None = triage_field("Phys Chan", verbose=1)
    rx_link_up: str | None = triage_field("RX Link Up", verbose=1)
    retrain_delta: int | None = triage_field("Retrain Delta", verbose=1)
    heartbeat_values: str | None = triage_field("Heartbeat Values", verbose=1)
    link_raw: int | None = triage_field("Link Raw", hex_serializer, verbose=2)
    sample_gap: float | None = triage_field("Sample Gap", verbose=2)
    mailbox_raw: str | None = triage_field("Mailbox Raw", verbose=2)

    def __init__(self):
        self.channel = None
        self.port_status = None
        self.link = None
        self.retrain_count = None
        self.heartbeat = None
        self.mailbox = None
        self.physical_channel = None
        self.rx_link_up = None
        self.retrain_delta = None
        self.heartbeat_values = None
        self.link_raw = None
        self.sample_gap = None
        self.mailbox_raw = None


@dataclass
class EthFinding:
    is_error: bool
    message: str


def _block_unreadable(words: dict[int, int], address: int, count: int) -> bool:
    return all(words[address + 4 * index] == ALL_ONES for index in range(count))


def _new_signatures(
    address: int, first: EthCoreSample | None, second: EthCoreSample, burst: list[dict[int, int]]
) -> set[int] | None:
    """Signatures of the values written to a heartbeat word after the first sample, or None if it can't tell."""
    if first is None or first.words[address] == ALL_ONES:
        return None
    before = first.words[address]
    later = [second.words[address]] + [reads[address] for reads in burst if address in reads]
    return {value >> 16 for value in later if value not in (before, ALL_ONES)}


def _heartbeat_text(signatures: list[set[int] | None], halted: bool) -> str:
    written: set[int] = set()
    for found in signatures:
        if found is None:
            return "One sample"
        written |= found
    writers = [name for signature, name in WRITER_NAMES if signature in written]
    if writers:
        return f"Moving ({', '.join(writers)})"
    if written:
        return "Moving"
    return "Frozen (halted by triage)" if halted else "Frozen"


def _values_text(address: int, first: EthCoreSample | None, second: EthCoreSample) -> str:
    after = f"{second.words[address]:#010x}"
    return after if first is None else f"{first.words[address]:#010x} -> {after}"


def _link_text(up_before: bool | None, up_after: bool, stale: bool) -> str:
    text = "Up" if up_after else "Down"
    if up_before is not None and up_before != up_after:
        text = f"{'Up' if up_before else 'Down'} -> {text}"
    return f"{text} (stale)" if stale else text


def _link_findings(up_before: bool | None, up_after: bool, fresh: bool, expected_up: bool) -> list[EthFinding]:
    if up_before is None:
        return []
    if up_before != up_after:
        return [EthFinding(False, LINK_CHANGED)]
    if up_after or not expected_up:
        return []
    return [EthFinding(True, LINK_DOWN)] if fresh else [EthFinding(False, LINK_DOWN_STALE)]


def _retrains(row: EthCoreCheckData, address: int, first: EthCoreSample | None, second: EthCoreSample) -> None:
    after = second.words[address]
    if after == ALL_ONES:
        return
    row.retrain_count = after
    if first is not None and first.words[address] != ALL_ONES:
        row.retrain_delta = (after - first.words[address]) & 0xFFFFFFFF


def _unreadable_row(row: EthCoreCheckData) -> tuple[EthCoreCheckData, list[EthFinding]]:
    row.port_status = row.link = row.heartbeat = "Unreadable"
    return row, [EthFinding(False, CORE_UNREADABLE)]


def _wormhole_port_status(words: dict[int, int], channel: int | None) -> str:
    disable_mask = words[WH_PORT_DISABLE_MASK]
    train_status = words[WH_TRAIN_STATUS]
    if ALL_ONES in (disable_mask, words[WH_RETRAIN_FORCE], train_status, words[WH_LINK_ERROR_STATUS]):
        return "Unreadable"
    if channel is not None and (disable_mask >> channel) & 1:
        return "Disabled"
    if words[WH_RETRAIN_FORCE] == 1:
        return "Retraining"
    if train_status == 2 and words[WH_LINK_ERROR_STATUS] >= WH_NOT_CONNECTED_ERROR_CODE:
        return "Not connected"
    return WH_TRAIN_STATUS_NAMES.get(train_status, "Invalid")


def decode_wormhole_core(
    channel: int | None,
    first: EthCoreSample | None,
    second: EthCoreSample,
    burst: list[dict[int, int]],
    halted: bool,
) -> tuple[EthCoreCheckData, list[EthFinding]]:
    row = EthCoreCheckData()
    row.channel = channel
    words = second.words
    if _block_unreadable(words, WH_BOOT_RESULTS, WH_BOOT_RESULTS_WORDS):
        return _unreadable_row(row)
    if first is not None and _block_unreadable(first.words, WH_BOOT_RESULTS, WH_BOOT_RESULTS_WORDS):
        first = None
    findings: list[EthFinding] = []

    row.physical_channel = words[WH_LOCAL_ETH_ID]
    row.port_status = _wormhole_port_status(words, channel)
    if row.port_status in WH_NOT_TRAINED_STATES:
        findings.append(EthFinding(False, PORT_NOT_TRAINED))
    elif row.port_status == "Invalid":
        findings.append(EthFinding(False, PORT_STATUS_INVALID))

    # 0x1C moves only when base FW runs its loop, which is also when it refreshes link_status.
    base_fw = _new_signatures(WH_HEARTBEAT, first, second, burst)
    shared = _new_signatures(WH_SHARED_HEARTBEAT, first, second, burst)
    base_fw_moving = base_fw is not None and BASE_FW_SIGNATURE in base_fw
    row.heartbeat = _heartbeat_text([base_fw, shared], halted)
    row.heartbeat_values = (
        f"0x1C {_values_text(WH_HEARTBEAT, first, second)}, "
        f"0x1F80 {_values_text(WH_SHARED_HEARTBEAT, first, second)}"
    )
    # Without metal firmware on the core, nothing stops base FW from looping.
    no_metal_firmware = (
        first is not None and words[WH_LAUNCH_ERISC_APP_FLAG] == 0 and first.words[WH_LAUNCH_ERISC_APP_FLAG] == 0
    )
    if base_fw == set() and no_metal_firmware and not halted:
        findings.append(EthFinding(False, BASE_FW_NOT_PROGRESSING))

    row.link_raw = words[WH_LINK_STATUS]
    up_after = words[WH_LINK_STATUS] == WH_LINK_UP
    up_before = None if first is None else first.words[WH_LINK_STATUS] == WH_LINK_UP
    row.rx_link_up = "Up" if up_after else "Down"
    row.link = _link_text(up_before, up_after, stale=first is not None and not base_fw_moving)
    findings += _link_findings(
        up_before, up_after, fresh=base_fw_moving, expected_up=row.port_status not in ("Disabled", "Not connected")
    )

    _retrains(row, WH_RETRAIN_COUNT, first, second)
    if first is not None:
        row.sample_gap = round(second.timestamp - first.timestamp, 2)
    return row, findings


def _mailbox_text(words: dict[int, int]) -> str:
    slots = []
    for index, name in enumerate(MAILBOX_SLOT_NAMES):
        message = words[BH_MAILBOX + index * BH_MAILBOX_SLOT_SIZE]
        if message == 0:
            continue
        status = MAILBOX_STATUS_NAMES.get(message >> 16, f"{message >> 16:#06x}")
        kind = MAILBOX_TYPE_NAMES.get(message & 0xFFFF, f"{message & 0xFFFF:#x}")
        slots.append(f"{name}: {status} {kind}")
    return ", ".join(slots) if slots else "Idle"


def _mailbox_pending_in_both(first: dict[int, int], second: dict[int, int]) -> bool:
    for index in range(BH_MAILBOX_SLOTS):
        address = BH_MAILBOX + index * BH_MAILBOX_SLOT_SIZE
        if first[address] == second[address] and second[address] >> 16 in (MAILBOX_CALL, MAILBOX_CALL_ACK):
            return True
    return False


def decode_blackhole_core(
    channel: int | None,
    first: EthCoreSample | None,
    second: EthCoreSample,
    burst: list[dict[int, int]],
    halted: bool,
) -> tuple[EthCoreCheckData, list[EthFinding]]:
    row = EthCoreCheckData()
    row.channel = channel
    words = second.words
    if _block_unreadable(words, BH_BOOT_RESULTS, BH_BOOT_RESULTS_WORDS):
        return _unreadable_row(row)
    if first is not None and _block_unreadable(first.words, BH_BOOT_RESULTS, BH_BOOT_RESULTS_WORDS):
        first = None
    findings: list[EthFinding] = []

    row.physical_channel = (words[BH_LOCAL_INFO] >> 16) & 0xFF
    port = words[BH_PORT_STATUS]
    row.port_status = "Unreadable" if port == ALL_ONES else BH_PORT_STATUS_NAMES.get(port, "Invalid")
    if row.port_status == "Unknown":
        findings.append(EthFinding(False, PORT_NOT_TRAINED))
    elif row.port_status == "Invalid":
        findings.append(EthFinding(False, PORT_STATUS_INVALID))

    rx_link_up = words[BH_RX_LINK_UP]
    row.rx_link_up = "Unreadable" if rx_link_up == ALL_ONES else ("Up" if rx_link_up == 1 else "Down")
    pcs_after = words[BH_PCS_STATUS]
    pcs_before = None if first is None else first.words[BH_PCS_STATUS]
    if ALL_ONES not in (pcs_after, pcs_before):
        # PCS_STATUS is live; port_status and rx_link_up are snapshots base FW may not have refreshed.
        row.link_raw = pcs_after
        up_before = None if pcs_before is None else pcs_before == 1
        row.link = _link_text(up_before, pcs_after == 1, stale=False)
        findings += _link_findings(up_before, pcs_after == 1, fresh=True, expected_up=row.port_status != "Unused")
    elif rx_link_up == ALL_ONES:
        row.link = "Unreadable"
    else:
        up_before = None if first is None else first.words[BH_RX_LINK_UP] == 1
        row.link = _link_text(up_before, rx_link_up == 1, stale=True)
        findings += _link_findings(up_before, rx_link_up == 1, fresh=False, expected_up=row.port_status != "Unused")

    signatures = _new_signatures(BH_HEARTBEAT, first, second, burst)
    row.heartbeat = _heartbeat_text([signatures], halted)
    row.heartbeat_values = _values_text(BH_HEARTBEAT, first, second)

    row.mailbox = _mailbox_text(words)
    row.mailbox_raw = " ".join(
        f"{words[BH_MAILBOX + 4 * index]:#010x}" for index in range(BH_MAILBOX_SLOTS * BH_MAILBOX_SLOT_SIZE // 4)
    )
    if first is not None and _mailbox_pending_in_both(first.words, words):
        findings.append(EthFinding(False, MAILBOX_NOT_SERVICED))

    _retrains(row, BH_RETRAIN_COUNT, first, second)
    if first is not None:
        row.sample_gap = round(second.timestamp - first.timestamp, 2)
    return row, findings


def _logical_channel(location: OnChipCoordinate) -> int | None:
    try:
        return int(location.to("logical")[0][1])
    except Exception:
        return None


def get_eth_core_data(location: OnChipCoordinate, sampling: EthStatusSampling) -> EthCoreCheckData | None:
    device = location.device
    plan = get_read_plan(location)
    if plan is None:
        utils.ERROR(f"Unsupported architecture for check_eth_status: {device._arch}")
        return None
    channel = _logical_channel(location)
    first = sampling.get_initial_sample(location)
    try:
        second = sampling.read_sample(location)
        assert second is not None
        burst: list[dict[int, int]] = []
        if first is not None and any(first.words[address] == second.words[address] for address in plan.heartbeats):
            burst = sampling.read_heartbeat_burst(location)
    except Exception as e:
        log_warning_location(location, f"{CORE_UNREADABLE}: {e}")
        row = EthCoreCheckData()
        row.channel = channel
        return _unreadable_row(row)[0]

    session = get_triage_session()
    halted = any(session.is_halted_core(location, risc_name) for risc_name in location.noc_block.risc_names)
    decode = decode_wormhole_core if device.is_wormhole() else decode_blackhole_core
    row, findings = decode(channel, first, second, burst, halted)
    for finding in findings:
        if finding.is_error:
            log_check_location(location, False, finding.message)
        else:
            log_warning_location(location, finding.message)
    return row


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    sampling = get_eth_status_sampling(args, context)
    sampling.wait_for_second_sample()
    return run_checks.run_per_block_check(
        lambda location: get_eth_core_data(location, sampling), block_filter=["active_eth"]
    )


if __name__ == "__main__":
    run_script()
