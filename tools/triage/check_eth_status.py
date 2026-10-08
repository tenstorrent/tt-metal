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
    Reads base-firmware state once and reports port and link state, retrain, CRC and FEC counts, the firmware
    signature and, on Blackhole, ERR_STAT, the TX resend and RX drop totals and the raw eth firmware mailbox messages.
    The heartbeat is read on every core first and compared after a single 100 ms wait; one that has not moved is
    reported Down, as a warning.
    A link that is down on a port expected to be up is an error. Counters are reported but never flagged.
    Cores running eth firmware older than UMD supports show raw values only, with no checks.

Owner:
    nhuang-tt
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import struct
import time
from typing import cast

from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, log_check_location, log_warning_location, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device
from ttexalens.umd_device import TimeoutDeviceRegisterError
import utils

script_config = ScriptConfig(
    depends=["run_checks"],
)


@dataclass
class EthCoreCheckData:
    port_status: str | None = triage_field("Port Status")
    link: str | None = triage_field("Link")
    heartbeat: str | None = triage_field("Heartbeat")
    retrain_count: int | None = triage_field("Retrain Count")
    crc_errors: int | None = triage_field("CRC Errors")
    corrected_codewords: int | None = triage_field("Corr CW")
    uncorrected_codewords: int | None = triage_field("Uncorr CW")
    rx_link_up: str | None = triage_field("RX Link Up")
    err_stat: int | None = triage_field("ERR_STAT", hex_serializer)
    txq_resends: int | None = triage_field("TXQ Resends")
    rxq_drops: int | None = triage_field("RXQ Drops")
    eth_fw: str | None = triage_field("ETH FW")
    fw_signature: int | None = triage_field("FW Signature", hex_serializer, verbose=1)  # 0xABCD base FW, 0xDCBA router
    link_raw: int | None = triage_field("Link Raw", hex_serializer, verbose=1)
    mailbox_host: int | None = triage_field("Mailbox HOST", hex_serializer, verbose=1)
    mailbox_risc1: int | None = triage_field("Mailbox RISC1", hex_serializer, verbose=1)
    mailbox_cmfw: int | None = triage_field("Mailbox CMFW", hex_serializer, verbose=1)
    mailbox_other: int | None = triage_field("Mailbox OTHER", hex_serializer, verbose=1)

    def __init__(self):
        for field in fields(self):
            setattr(self, field.name, None)


def _u64(words: dict[int, int], address: int) -> int:
    return (words[address + 4] << 32) | words[address]


class EthCore(ABC):
    """
    Base class for Ethernet cores that provides common functionality.
    """

    # Set by each architecture.
    BOOT_RESULTS: int  # boot_results_t, read in one access
    BOOT_RESULTS_WORDS: int
    EXTRA_WORDS: tuple[int, ...]  # read one at a time
    HEARTBEAT: int  # in the block, written by base FW and the fabric router

    def __init__(self, location: OnChipCoordinate, context: Context):
        self.location = location
        self.context = context
        self.first_heartbeat: int | None = None  # read in run(), at least 100 ms before get_results

    @abstractmethod
    def decode(self, row: EthCoreCheckData, words: dict[int, int]) -> None:
        """Fills the row from the words read and logs the checks."""

    def get_results(self) -> EthCoreCheckData:
        """Get and log all ethernet core status results."""
        row = EthCoreCheckData()
        try:
            data = read_from_device(
                self.location, self.BOOT_RESULTS, num_bytes=4 * self.BOOT_RESULTS_WORDS, context=self.context
            )
            # A failed host read returns all ones.
            if data == b"\xff" * len(data):
                raise ValueError("all words read 0xFFFFFFFF")
            values = struct.unpack(f"<{self.BOOT_RESULTS_WORDS}I", data)
            words = {self.BOOT_RESULTS + 4 * index: value for index, value in enumerate(values)}
            for address in self.EXTRA_WORDS:
                words[address] = read_word_from_device(self.location, address, context=self.context)
        except TimeoutDeviceRegisterError:
            raise
        except Exception as e:
            log_warning_location(self.location, f"Eth core is unreadable: {e}")
            row.port_status = row.link = "Unreadable"
            return row
        self.decode(row, words)
        if self.first_heartbeat is not None and row.port_status != "Unsupported FW":
            row.heartbeat = "Down" if words[self.HEARTBEAT] == self.first_heartbeat else "Up"
            if row.heartbeat == "Down":
                log_warning_location(self.location, "Eth heartbeat is down")
        return row


class WormholeEthCore(EthCore):
    """Wormhole-specific Ethernet core implementation."""

    # Base firmware: wormhole/eth_fw_api.h, eth_l1_address_map.h, UMD wormhole_eth.hpp.
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
    HEARTBEAT = 0x1F80  # base FW (0xABCD) and fabric router (0xDCBA)
    CORRECTED_CODEWORDS = 0x1F90  # FEC corr_cw: high word, then low word
    UNCORRECTED_CODEWORDS = 0x1F98
    EXTRA_WORDS = (ETH_FW_VERSION, PORT_DISABLE_MASK, TRAIN_STATUS, LINK_ERROR_STATUS)

    # Oldest eth FW these addresses hold for (UMD's minimum ERISC FW version).
    MIN_ETH_FW_VERSION = (6, 14, 0)

    TRAIN_STATUS_NAMES = {0: "Training", 1: "Trained", 2: "Train failed"}
    # Link error codes from here up mean nothing is plugged in (UMD ETH_LINK_UNUSED_ERROR_CODE_RANGE_START).
    NOT_CONNECTED_ERROR_CODE = 11

    def decode(self, row: EthCoreCheckData, words: dict[int, int]) -> None:
        row.link_raw = words[self.LINK_STATUS]
        version_word = words[self.ETH_FW_VERSION]
        version = (version_word >> 16) & 0xFF, (version_word >> 12) & 0xF, version_word & 0xFFF
        row.eth_fw = ".".join(str(part) for part in version)
        # Older FW may lay L1 out differently, so only raw values are kept. No FW writes a version of 0.
        if version_word and version < self.MIN_ETH_FW_VERSION:
            row.port_status = "Unsupported FW"
            return

        channel = self.location.to("logical")[0][1]
        train_status = words[self.TRAIN_STATUS]
        if (words[self.PORT_DISABLE_MASK] >> channel) & 1:
            row.port_status = "Disabled"
        elif words[self.RETRAIN_FORCE] == 1:
            row.port_status = "Retraining"
        elif train_status == 2 and words[self.LINK_ERROR_STATUS] >= self.NOT_CONNECTED_ERROR_CODE:
            row.port_status = "Not connected"
        else:
            row.port_status = self.TRAIN_STATUS_NAMES.get(train_status, "Invalid")
        row.link = "Up" if row.link_raw == self.LINK_UP else "Down"
        if row.port_status != "Trained":
            log_warning_location(self.location, "Eth port is not trained")
        log_check_location(
            self.location, row.link == "Up" or row.port_status in ("Disabled", "Not connected"), "Eth link is down"
        )

        row.fw_signature = words[self.HEARTBEAT] >> 16
        row.retrain_count = words[self.RETRAIN_COUNT]
        row.crc_errors = words[self.CRC_ERRORS]
        row.corrected_codewords = (words[self.CORRECTED_CODEWORDS] << 32) | words[self.CORRECTED_CODEWORDS + 4]
        row.uncorrected_codewords = (words[self.UNCORRECTED_CODEWORDS] << 32) | words[self.UNCORRECTED_CODEWORDS + 4]


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
    MAILBOX = 0x7D000  # slots HOST, RISC1, CMFW, OTHER
    MAILBOX_SLOT_SIZE = 16  # msg + 3 args
    PCS_STATUS = 0xFFB9800C  # live link status, up only when it equals 1
    ERR_STAT = 0xFFB980D8  # latched ETH_CTRL error status; reading does not clear it
    EXTRA_WORDS = (PCS_STATUS, ERR_STAT)

    # Oldest eth FW these addresses hold for (UMD's minimum ERISC FW version).
    MIN_ETH_FW_VERSION = (1, 4, 1)
    # First eth FW that writes the TXQ/RXQ counters.
    QUEUE_COUNTERS_MIN_ETH_FW_VERSION = (1, 5, 0)

    PORT_STATUS_NAMES = {0: "Unknown", 1: "Up", 2: "Down", 3: "Unused"}

    def decode(self, row: EthCoreCheckData, words: dict[int, int]) -> None:
        row.mailbox_host = words[self.MAILBOX]
        row.mailbox_risc1 = words[self.MAILBOX + self.MAILBOX_SLOT_SIZE]
        row.mailbox_cmfw = words[self.MAILBOX + 2 * self.MAILBOX_SLOT_SIZE]
        row.mailbox_other = words[self.MAILBOX + 3 * self.MAILBOX_SLOT_SIZE]
        row.err_stat = words[self.ERR_STAT]
        # PCS_STATUS is a live register; port_status and rx_link_up are snapshots base FW may not have refreshed.
        row.link_raw = words[self.PCS_STATUS]
        row.link = "Up" if row.link_raw == 1 else "Down"
        version_word = words[self.ETH_FW_VERSION]
        version = (version_word >> 16) & 0xFF, (version_word >> 8) & 0xFF, version_word & 0xFF
        row.eth_fw = ".".join(str(part) for part in version)
        # Older FW may lay L1 out differently, so only raw values are kept. No FW writes a version of 0.
        if version_word and version < self.MIN_ETH_FW_VERSION:
            row.port_status = "Unsupported FW"
            return

        row.port_status = self.PORT_STATUS_NAMES.get(words[self.PORT_STATUS], "Invalid")
        if row.port_status in ("Unknown", "Invalid"):
            log_warning_location(self.location, "Eth port is not trained")
        log_check_location(self.location, row.link == "Up" or row.port_status == "Unused", "Eth link is down")

        row.rx_link_up = "Up" if words[self.RX_LINK_UP] == 1 else "Down"
        row.fw_signature = words[self.HEARTBEAT] >> 16
        row.retrain_count = words[self.RETRAIN_COUNT]
        row.crc_errors = _u64(words, self.RX_BAD_FCS)
        row.corrected_codewords = _u64(words, self.CORRECTED_CODEWORDS)
        row.uncorrected_codewords = _u64(words, self.UNCORRECTED_CODEWORDS)
        if not version_word or version >= self.QUEUE_COUNTERS_MIN_ETH_FW_VERSION:
            row.txq_resends = sum(_u64(words, self.TXQ_RESENDS + 8 * queue) for queue in range(self.QUEUES))
            row.rxq_drops = sum(_u64(words, self.RXQ_DROPS + 8 * queue) for queue in range(self.QUEUES))


def get_eth_core(location: OnChipCoordinate, context: Context) -> EthCore | None:
    """Create appropriate EthCore instance based on device type and take its first heartbeat read."""
    eth_core: EthCore
    if location.device.is_wormhole():
        eth_core = WormholeEthCore(location, context)
    elif location.device.is_blackhole():
        eth_core = BlackholeEthCore(location, context)
    else:
        utils.ERROR(f"Unsupported architecture for check_eth_status: {location.device._arch}")
        return None
    try:
        eth_core.first_heartbeat = read_word_from_device(location, eth_core.HEARTBEAT, context=context)
    except TimeoutDeviceRegisterError:
        raise
    except Exception:
        pass  # get_results reports the core if its own reads fail
    return eth_core


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    first_pass = run_checks.run_per_block_check(
        lambda location: get_eth_core(location, context), block_filter=["active_eth"]
    )
    eth_cores = {result.location: cast(EthCore, result.result) for result in first_pass or []}
    time.sleep(0.1)
    return run_checks.run_per_block_check(
        lambda location: eth_cores[location].get_results() if location in eth_cores else None,
        block_filter=["active_eth"],
    )


if __name__ == "__main__":
    run_script()
