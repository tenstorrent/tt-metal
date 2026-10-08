#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_eth_status.py

Description:
    Checks the link on every active ethernet core: the cores metal found trained when it started (from Inspector),
    or, without Inspector, the channels connected to another chip on this host when triage starts (links to other
    hosts are not checked).
    Reads base-firmware state once and reports port and link state, retrain, CRC and FEC counts, the heartbeat and
    firmware signature and, on Blackhole, ERR_STAT, the TX/RX queue counters and the eth firmware mailboxes.
    A link that is down on a port expected to be up is an error. Counters are reported but never flagged.
    Cores running eth firmware older than UMD supports show raw values only, with no findings.

Owner:
    nhuang-tt
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import struct

from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, log_check_location, log_warning_location, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device
import utils

script_config = ScriptConfig(
    depends=["run_checks"],
)

# What a failed host read returns.
ALL_ONES = 0xFFFFFFFF

# Finding messages carry no values, so sqlite can group by message; values stay in the columns.
LINK_DOWN = "Eth link is down"
PORT_NOT_TRAINED = "Eth port is not trained"
PORT_STATUS_INVALID = "Eth port status is invalid"
CORE_UNREADABLE = "Eth core is unreadable"


@dataclass
class EthCoreCheckData:
    channel: int | None = triage_field("Chan")
    port_status: str | None = triage_field("Port Status")
    link: str | None = triage_field("Link")
    retrain_count: int | None = triage_field("Retrain Count")
    crc_errors: int | None = triage_field("CRC Errors")
    corrected_codewords: int | None = triage_field("Corr CW")
    uncorrected_codewords: int | None = triage_field("Uncorr CW")
    mailbox: str | None = triage_field("Mailbox")
    eth_fw: str | None = triage_field("ETH FW", verbose=1)
    fw_signature: int | None = triage_field("FW Signature", hex_serializer, verbose=1)  # tt_eth_firmware_signature
    rx_link_up: str | None = triage_field("RX Link Up", verbose=1)
    err_stat: int | None = triage_field("ERR_STAT", hex_serializer, verbose=1)
    txq_resends: str | None = triage_field("TXQ Resends", verbose=1)
    rxq_drops: str | None = triage_field("RXQ Drops", verbose=1)
    heartbeat: str | None = triage_field("Heartbeat", verbose=2)
    link_raw: int | None = triage_field("Link Raw", hex_serializer, verbose=2)
    mailbox_raw: str | None = triage_field("Mailbox Raw", verbose=2)

    def __init__(self, channel: int | None):
        for field in fields(self):
            setattr(self, field.name, None)
        self.channel = channel


@dataclass
class EthFinding:
    is_error: bool
    message: str


def _unreadable_row(row: EthCoreCheckData) -> tuple[EthCoreCheckData, list[EthFinding]]:
    row.port_status = row.link = "Unreadable"
    return row, [EthFinding(False, CORE_UNREADABLE)]


def _unsupported_fw_row(row: EthCoreCheckData) -> tuple[EthCoreCheckData, list[EthFinding]]:
    # Older FW may lay base-FW L1 out differently, so only raw words and live registers are kept, with no findings.
    row.port_status = "Unsupported FW"
    return row, []


def _u64(words: dict[int, int], address: int, high_first: bool = False) -> int:
    first, second = words[address], words[address + 4]
    return (first << 32) | second if high_first else (second << 32) | first


def _signature(word: int) -> int | None:
    return None if word == ALL_ONES else word >> 16


def _logical_channel(location: OnChipCoordinate) -> int | None:
    try:
        return int(location.to("logical")[0][1])
    except Exception:
        return None


class EthCore(ABC):
    """
    Base class for Ethernet cores that provides common functionality.
    """

    # Set by each architecture.
    BOOT_RESULTS: int  # boot_results_t, read in one access
    BOOT_RESULTS_WORDS: int
    EXTRA_WORDS: tuple[int, ...]  # read one at a time
    ETH_FW_VERSION: int

    def __init__(self, location: OnChipCoordinate, context: Context):
        self.location = location
        self.context = context

    @abstractmethod
    def fw_version(self, word: int) -> tuple[int, int, int]:
        """Splits the eth FW version word into major, minor and patch."""

    @abstractmethod
    def decode(self, channel: int | None, words: dict[int, int]) -> tuple[EthCoreCheckData, list[EthFinding]]:
        """Fills a row from the words read and returns its findings."""

    def read(self) -> dict[int, int]:
        count = self.BOOT_RESULTS_WORDS
        data = read_from_device(self.location, self.BOOT_RESULTS, num_bytes=4 * count, context=self.context)
        words = {self.BOOT_RESULTS + 4 * index: value for index, value in enumerate(struct.unpack(f"<{count}I", data))}
        for address in self.EXTRA_WORDS:
            words[address] = read_word_from_device(self.location, address, context=self.context)
        return words

    def eth_fw_version(self, row: EthCoreCheckData, words: dict[int, int]) -> tuple[int, int, int] | None:
        """Fills the ETH FW column; None when the word holds no version (unreadable, or 0, which no FW writes)."""
        word = words[self.ETH_FW_VERSION]
        if word in (0, ALL_ONES):
            row.eth_fw = "Unreadable" if word == ALL_ONES else "Unknown"
            return None
        version = self.fw_version(word)
        row.eth_fw = ".".join(str(part) for part in version)
        return version

    def get_results(self) -> EthCoreCheckData:
        """Get and log all ethernet core status results."""
        channel = _logical_channel(self.location)
        try:
            words = self.read()
        except Exception as e:
            log_warning_location(self.location, f"{CORE_UNREADABLE}: {e}")
            return _unreadable_row(EthCoreCheckData(channel))[0]

        row, findings = self.decode(channel, words)
        for finding in findings:
            if finding.is_error:
                log_check_location(self.location, False, finding.message)
            else:
                log_warning_location(self.location, finding.message)
        return row


class WormholeEthCore(EthCore):
    """Wormhole-specific Ethernet core implementation."""

    # Base firmware: wormhole/eth_fw_api.h, eth_l1_address_map.h, UMD wormhole_eth.hpp.
    HEARTBEAT = 0x1C  # base FW (0xABCD) once per loop pass
    ETH_FW_VERSION = 0x210  # major in bits 23:16, minor 15:12, patch 11:0
    PORT_DISABLE_MASK = 0x1008  # bit N = channel N disabled
    TRAIN_STATUS = 0x1104  # 0 in progress, 1 success, 2 fail
    LINK_ERROR_STATUS = 0x1440  # training error code
    BOOT_RESULTS = 0x1EC0  # boot_results_t
    BOOT_RESULTS_WORDS = 80
    LINK_STATUS = 0x1ED4  # link is up only when it equals LINK_UP
    LINK_UP = 6
    RETRAIN_COUNT = 0x1EDC
    RETRAIN_FORCE = 0x1EFC  # 1 while a requested retrain is pending
    CRC_ERRORS = 0x1F7C
    SHARED_HEARTBEAT = 0x1F80  # base FW (0xABCD) and fabric router (0xDCBA)
    CORRECTED_CODEWORDS = 0x1F90  # FEC corr_cw: high word, then low word
    UNCORRECTED_CODEWORDS = 0x1F98
    EXTRA_WORDS = (HEARTBEAT, ETH_FW_VERSION, PORT_DISABLE_MASK, TRAIN_STATUS, LINK_ERROR_STATUS)

    # Oldest eth FW these addresses hold for (UMD's minimum ERISC FW version).
    MIN_ETH_FW_VERSION = (6, 14, 0)

    TRAIN_STATUS_NAMES = {0: "Training", 1: "Trained", 2: "Train failed"}
    # Link error codes from here up mean nothing is plugged in (UMD ETH_LINK_UNUSED_ERROR_CODE_RANGE_START).
    NOT_CONNECTED_ERROR_CODE = 11
    NOT_TRAINED_STATES = ("Disabled", "Retraining", "Training", "Train failed", "Not connected")

    def fw_version(self, word: int) -> tuple[int, int, int]:
        return (word >> 16) & 0xFF, (word >> 12) & 0xF, word & 0xFFF

    def _port_status(self, words: dict[int, int], channel: int | None) -> str:
        disable_mask = words[self.PORT_DISABLE_MASK]
        train_status = words[self.TRAIN_STATUS]
        if ALL_ONES in (disable_mask, train_status, words[self.LINK_ERROR_STATUS]):
            return "Unreadable"
        if channel is not None and (disable_mask >> channel) & 1:
            return "Disabled"
        if words[self.RETRAIN_FORCE] == 1:
            return "Retraining"
        if train_status == 2 and words[self.LINK_ERROR_STATUS] >= self.NOT_CONNECTED_ERROR_CODE:
            return "Not connected"
        return self.TRAIN_STATUS_NAMES.get(train_status, "Invalid")

    def decode(self, channel: int | None, words: dict[int, int]) -> tuple[EthCoreCheckData, list[EthFinding]]:
        row = EthCoreCheckData(channel)
        if all(words[self.BOOT_RESULTS + 4 * index] == ALL_ONES for index in range(self.BOOT_RESULTS_WORDS)):
            return _unreadable_row(row)
        row.link_raw = words[self.LINK_STATUS]
        row.heartbeat = f"0x1C {words[self.HEARTBEAT]:#010x}, 0x1F80 {words[self.SHARED_HEARTBEAT]:#010x}"
        version = self.eth_fw_version(row, words)
        if version is not None and version < self.MIN_ETH_FW_VERSION:
            return _unsupported_fw_row(row)
        findings: list[EthFinding] = []

        row.port_status = self._port_status(words, channel)
        if row.port_status in self.NOT_TRAINED_STATES:
            findings.append(EthFinding(False, PORT_NOT_TRAINED))
        elif row.port_status == "Invalid":
            findings.append(EthFinding(False, PORT_STATUS_INVALID))
        row.link = "Up" if words[self.LINK_STATUS] == self.LINK_UP else "Down"
        if row.link == "Down" and row.port_status not in ("Disabled", "Not connected"):
            findings.append(EthFinding(True, LINK_DOWN))

        row.fw_signature = _signature(words[self.HEARTBEAT])
        row.retrain_count = words[self.RETRAIN_COUNT]
        row.crc_errors = words[self.CRC_ERRORS]
        row.corrected_codewords = _u64(words, self.CORRECTED_CODEWORDS, high_first=True)
        row.uncorrected_codewords = _u64(words, self.UNCORRECTED_CODEWORDS, high_first=True)
        return row, findings


class BlackholeEthCore(EthCore):
    """Blackhole-specific Ethernet core implementation."""

    # Base firmware: blackhole/eth_fw_api.h, 32-bit device layout.
    BOOT_RESULTS = 0x7CC00  # boot_results_t (1 KB), then the eth FW mailboxes (64 B)
    BOOT_RESULTS_WORDS = 272
    PORT_STATUS = 0x7CC04
    HEARTBEAT = 0x7CC70  # heartbeat[0]: base FW (0xABCD) and fabric router (0xDCBA)
    RETRAIN_COUNT = 0x7CE00
    RX_LINK_UP = 0x7CE04  # snapshot, up only when it equals 1
    # The counters below are u64, low word first.
    RX_BAD_FCS = 0x7CE60  # frames_rxd_badfcs: received frames that failed the Ethernet CRC
    CORRECTED_CODEWORDS = 0x7CE90
    UNCORRECTED_CODEWORDS = 0x7CE98
    TXQ_RESENDS = 0x7CEA0  # txq0..2_resend_cnt
    RXQ_DROPS = 0x7CEB8  # rxq0..2_pkt_drop
    QUEUES = 3
    ETH_FW_VERSION = 0x7CFBC  # fw_version_t: patch, minor and major bytes
    MAILBOX = 0x7D000  # HOST, RISC1, CMFW, OTHER
    MAILBOX_SLOTS = 4
    MAILBOX_SLOT_SIZE = 16  # msg + 3 args
    PCS_STATUS = 0xFFB9800C  # live link status, up only when it equals 1
    ERR_STAT = 0xFFB980D8  # latched ETH_CTRL error status; reading does not clear it
    EXTRA_WORDS = (PCS_STATUS, ERR_STAT)

    # Oldest eth FW these addresses hold for (UMD's minimum ERISC FW version).
    MIN_ETH_FW_VERSION = (1, 4, 1)
    # First eth FW that writes the TXQ/RXQ counters.
    QUEUE_COUNTERS_MIN_ETH_FW_VERSION = (1, 5, 0)

    PORT_STATUS_NAMES = {0: "Unknown", 1: "Up", 2: "Down", 3: "Unused"}

    MAILBOX_SLOT_NAMES = ("HOST", "RISC1", "CMFW", "OTHER")
    MAILBOX_CALL = 0xCA11  # posted, not picked up yet
    MAILBOX_CALL_ACK = 0xCEDE  # picked up, handler still running (eth FW >= 1.5.0)
    MAILBOX_DONE = 0xD0E5
    MAILBOX_STATUS_NAMES = {MAILBOX_CALL: "CALL", MAILBOX_CALL_ACK: "CALL_ACK", MAILBOX_DONE: "DONE"}
    MAILBOX_TYPE_NAMES = {
        0x1: "LINK_STATUS_CHECK",
        0x2: "RELEASE_CORE",
        0x6: "PORT_REINIT_MACPCS",
        0x9: "PORT_ACTION",
        0xF: "DYNAMIC_NOC_INIT",
    }

    def fw_version(self, word: int) -> tuple[int, int, int]:
        return (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF

    def _queue_counts(self, words: dict[int, int], address: int) -> str:
        return "/".join(str(_u64(words, address + 8 * queue)) for queue in range(self.QUEUES))

    def _mailbox_text(self, words: dict[int, int]) -> str:
        slots = []
        for index, name in enumerate(self.MAILBOX_SLOT_NAMES):
            message = words[self.MAILBOX + index * self.MAILBOX_SLOT_SIZE]
            if message == 0:
                continue
            status = self.MAILBOX_STATUS_NAMES.get(message >> 16, f"{message >> 16:#06x}")
            kind = self.MAILBOX_TYPE_NAMES.get(message & 0xFFFF, f"{message & 0xFFFF:#x}")
            slots.append(f"{name}: {status} {kind}")
        return ", ".join(slots) if slots else "Idle"

    def decode(self, channel: int | None, words: dict[int, int]) -> tuple[EthCoreCheckData, list[EthFinding]]:
        row = EthCoreCheckData(channel)
        if all(words[self.BOOT_RESULTS + 4 * index] == ALL_ONES for index in range(self.BOOT_RESULTS_WORDS)):
            return _unreadable_row(row)
        row.heartbeat = f"{words[self.HEARTBEAT]:#010x}"
        row.mailbox_raw = " ".join(
            f"{words[self.MAILBOX + 4 * index]:#010x}"
            for index in range(self.MAILBOX_SLOTS * self.MAILBOX_SLOT_SIZE // 4)
        )
        row.err_stat = None if words[self.ERR_STAT] == ALL_ONES else words[self.ERR_STAT]
        # PCS_STATUS is a live register; port_status and rx_link_up are snapshots base FW may not have refreshed.
        pcs_status = words[self.PCS_STATUS]
        row.link = "Unreadable" if pcs_status == ALL_ONES else ("Up" if pcs_status == 1 else "Down")
        if pcs_status != ALL_ONES:
            row.link_raw = pcs_status
        version = self.eth_fw_version(row, words)
        if version is not None and version < self.MIN_ETH_FW_VERSION:
            return _unsupported_fw_row(row)
        findings: list[EthFinding] = []

        row.port_status = self.PORT_STATUS_NAMES.get(words[self.PORT_STATUS], "Invalid")
        if row.port_status == "Unknown":
            findings.append(EthFinding(False, PORT_NOT_TRAINED))
        elif row.port_status == "Invalid":
            findings.append(EthFinding(False, PORT_STATUS_INVALID))
        if row.link == "Down" and row.port_status != "Unused":
            findings.append(EthFinding(True, LINK_DOWN))
        row.rx_link_up = "Up" if words[self.RX_LINK_UP] == 1 else "Down"

        row.fw_signature = _signature(words[self.HEARTBEAT])
        row.mailbox = self._mailbox_text(words)
        row.retrain_count = words[self.RETRAIN_COUNT]
        row.crc_errors = _u64(words, self.RX_BAD_FCS)
        row.corrected_codewords = _u64(words, self.CORRECTED_CODEWORDS)
        row.uncorrected_codewords = _u64(words, self.UNCORRECTED_CODEWORDS)
        if version is None or version >= self.QUEUE_COUNTERS_MIN_ETH_FW_VERSION:
            row.txq_resends = self._queue_counts(words, self.TXQ_RESENDS)
            row.rxq_drops = self._queue_counts(words, self.RXQ_DROPS)
        return row, findings


def get_eth_core_data(location: OnChipCoordinate, context: Context) -> EthCoreCheckData | None:
    """Create appropriate EthCore instance based on device type and get results."""
    eth_core: EthCore
    if location.device.is_wormhole():
        eth_core = WormholeEthCore(location, context)
    elif location.device.is_blackhole():
        eth_core = BlackholeEthCore(location, context)
    else:
        utils.ERROR(f"Unsupported architecture for check_eth_status: {location.device._arch}")
        return None
    return eth_core.get_results()


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    return run_checks.run_per_block_check(
        lambda location: get_eth_core_data(location, context), block_filter=["active_eth"]
    )


if __name__ == "__main__":
    run_script()
