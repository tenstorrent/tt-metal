#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_eth_status.py

Description:
    This script checks the link is up and there are no retrain counts on active ethernet cores.
    An ethernet core is considered active if the port status is up.
    A link being down or high retrain counts may indicate the eth connection is unstable.

Owner:
    nhuang-tt
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import monotonic, sleep
from run_checks import run as get_run_checks
from triage import ScriptConfig, triage_field, log_check_location, run_script
from ttexalens import read_word_from_device
from ttexalens.context import Context
from ttexalens.device import Device, OnChipCoordinate
import utils

script_config = ScriptConfig(
    depends=["run_checks"],
)

# UMD v0.9.9 allows 50 ms for progress. Its former 5 ms limit rejected some running ETH cores.
# https://github.com/tenstorrent/tt-umd/blob/v0.9.9/device/api/umd/device/utils/timeouts.hpp
# Allow one window for a first valid sample, then another for progress.
# A late first valid sample can use both windows: 50 + 50 = 100 ms.
# The waits overlap across cores. Reads and scheduling can extend the elapsed time.
HEARTBEAT_TIMEOUT_SECONDS = 0.05
# Local polling interval, not the firmware update period. Avoid a busy read loop.
HEARTBEAT_POLL_INTERVAL_SECONDS = 0.001


@dataclass
class EthCoreDefinitions:
    """Arch specific addresses of various ethernet core fields"""

    port_status: int | None
    retrain_count: int
    rx_link_up: int
    heartbeat: int
    mailbox: int | None
    mailbox_slots: int


@dataclass
class EthCoreCheckData:
    port_status: str | None = triage_field("Port Status")
    retrain_count: int | None = triage_field("Retrain Count")
    rx_link_up: str | None = triage_field("RX Link Up")
    heartbeat: bool | None = triage_field("Heartbeat")
    mailbox: list[str] | None = triage_field("Mailbox")

    def __init__(self):
        self.port_status = None
        self.retrain_count = None
        self.rx_link_up = None
        self.heartbeat = None
        self.mailbox = None


class InvalidHeartbeatSignature(ValueError):
    """A nonzero heartbeat word has an unsupported firmware signature."""


class EthCore(ABC):
    """
    Base class for Ethernet cores that provides common functionality.
    """

    eth_core_definitions: EthCoreDefinitions
    BASE_FW_HEARTBEAT_SIGNATURE: int
    FABRIC_HEARTBEAT_SIGNATURE: int
    # Both supported architectures put the signature in bits 31:16.
    HEARTBEAT_SIGNATURE_SHIFT = 16

    def __init__(self, location: OnChipCoordinate, context: Context):
        self.location = location
        self.context = context
        self._heartbeat_sample: int | None = None
        self._heartbeat_deadline: float | None = None
        self.heartbeat_result: bool | None = None

    @abstractmethod
    def port_status_to_string(self, port_status: int) -> str | None:
        """Convert port status value to a readable string."""
        pass

    def is_valid_heartbeat(self, value: int) -> bool:
        """Check the heartbeat format for this architecture."""
        return value >> self.HEARTBEAT_SIGNATURE_SHIFT in (
            self.BASE_FW_HEARTBEAT_SIGNATURE,
            self.FABRIC_HEARTBEAT_SIGNATURE,
        )

    def read_heartbeat(self) -> int | None:
        """Read a valid heartbeat sample, or None if the word is still zero."""
        value = read_word_from_device(self.location, self.eth_core_definitions.heartbeat, context=self.context)
        if value == 0:
            return None
        if not self.is_valid_heartbeat(value):
            raise InvalidHeartbeatSignature(f"Invalid heartbeat signature: 0x{value:08X}")
        return value

    def poll_heartbeat(self) -> None:
        """Take one sample. Leave heartbeat_result as None while the check is pending."""
        try:
            read_data = self.read_heartbeat()
        except InvalidHeartbeatSignature as error:
            # Fail only this core. An exception that reaches RunChecks skips the rest of the device.
            log_check_location(self.location, False, str(error))
            self.heartbeat_result = False
            return

        now = monotonic()
        if self._heartbeat_deadline is None:
            # Bound the wait for a first valid sample, including cores that keep reading zero.
            self._heartbeat_deadline = now + HEARTBEAT_TIMEOUT_SECONDS
        if read_data is not None:
            if self._heartbeat_sample is None:
                self._heartbeat_sample = read_data
                # A late first valid sample needs a full window to prove progress.
                self._heartbeat_deadline = now + HEARTBEAT_TIMEOUT_SECONDS
            elif read_data != self._heartbeat_sample:
                self.heartbeat_result = True
                return

        # Zero is not a baseline. It must not restart the deadline or erase a valid sample.
        # Read before checking the deadline, since other cores can delay this poll.
        if now >= self._heartbeat_deadline:
            log_check_location(self.location, False, "No heartbeat detected")
            self.heartbeat_result = False

    def get_results(self) -> EthCoreCheckData:
        """Get and log all ethernet core status results."""
        output = EthCoreCheckData()
        # HEARTBEAT
        output.heartbeat = self.heartbeat_result

        # PORT STATUS
        if self.eth_core_definitions.port_status is not None:
            port_status = read_word_from_device(
                self.location, self.eth_core_definitions.port_status, context=self.context
            )
            port_status_str = self.port_status_to_string(port_status)
            if port_status_str is None:
                output.port_status = "Unknown"
            else:
                output.port_status = port_status_str
            log_check_location(self.location, port_status_str != "Down", "port is down")

        # if the port is unused the rest of these checks are not relevant
        if output.port_status in ("Unused", "Unknown", "Undefined", None):
            return output

        # RETRAIN COUNT
        output.retrain_count = int(
            read_word_from_device(self.location, self.eth_core_definitions.retrain_count, context=self.context)
        )
        log_check_location(
            self.location,
            not output.retrain_count,
            f"retrain count is {output.retrain_count}",
        )

        # RX LINK UP
        output.rx_link_up = (
            "Up"
            if read_word_from_device(self.location, self.eth_core_definitions.rx_link_up, context=self.context)
            else "Down"
        )
        log_check_location(
            self.location,
            output.rx_link_up == "Up",
            f"rx link is not up: {output.rx_link_up}",
        )

        # MAILBOX
        if self.eth_core_definitions.mailbox is not None:
            output.mailbox = []
            any_pending_message = False
            for i in range(self.eth_core_definitions.mailbox_slots):
                # Format each mailbox value as a hex string
                mailbox_value = read_word_from_device(
                    self.location, self.eth_core_definitions.mailbox + i * 4, context=self.context
                )
                output.mailbox.append(f"0x{mailbox_value:08X}")
                if mailbox_value & 0xFFFF0000 == 0xCA110000:
                    any_pending_message = True
                log_check_location(
                    self.location,
                    not any_pending_message,
                    f"mailbox: {output.mailbox} (pending message)",
                )
        else:
            output.mailbox = ["None"]

        return output


class WormholeEthCore(EthCore):
    """Wormhole-specific Ethernet core implementation."""

    # L1[0x1C]: bits 31:16 identify the firmware; bits 15:0 are its counter.
    # UMD 0.9.9 defines these C++ constants but does not export them to Python:
    # https://github.com/tenstorrent/tt-umd/blob/v0.9.9/device/api/umd/device/firmware/erisc_firmware.hpp
    # RISC_POST_HEARTBEAT in tt_metal/hw/inc/api/dataflow/dataflow_api.h writes the fabric format.
    BASE_FW_HEARTBEAT_SIGNATURE = 0xABCD
    FABRIC_HEARTBEAT_SIGNATURE = 0xAABB

    def __init__(self, location: OnChipCoordinate, context: Context):
        super().__init__(location, context)
        self.eth_core_definitions = EthCoreDefinitions(
            port_status=None,
            retrain_count=0x1EC0 + 0x28,
            rx_link_up=0x1EC0 + 0x20,
            heartbeat=0x1C,
            mailbox=None,
            mailbox_slots=0,
        )

    def port_status_to_string(self, port_status: int) -> str | None:
        """Convert Wormhole port status to readable string."""
        # Undefined right now for Wormhole. Need to find the mapping
        status_map = {0: "Undefined", 1: "Undefined", 2: "Undefined", 3: "Undefined"}
        return status_map.get(port_status, None)


class BlackholeEthCore(EthCore):
    """Blackhole-specific Ethernet core implementation."""

    # L1[0x7CC70] is heartbeat[0]: a signature in bits 31:16 and counter in bits 15:0.
    # The base firmware format is documented in blackhole/eth_fw_api.h::aerisc_context_switch.
    # fabric_erisc_router.cpp writes 0xDCBA0000 | fabric_heartbeat_counter to this address.
    # This differs from Wormhole's 0xAABB fabric signature; UMD does not export it to Python.
    BASE_FW_HEARTBEAT_SIGNATURE = 0xABCD
    FABRIC_HEARTBEAT_SIGNATURE = 0xDCBA

    def __init__(self, location: OnChipCoordinate, context: Context):
        super().__init__(location, context)
        self.eth_core_definitions = EthCoreDefinitions(
            port_status=0x7CC04,
            retrain_count=0x7CE00,
            rx_link_up=0x7CE04,
            heartbeat=0x7CC70,
            mailbox=0x7D000,
            mailbox_slots=4,
        )

    def port_status_to_string(self, port_status: int) -> str | None:
        """Convert Blackhole port status to readable string."""
        status_map: dict[int, str | None] = {0: None, 1: "Up", 2: "Down", 3: "Unused"}
        return status_map.get(port_status, None)


def get_eth_core(device: Device, location: OnChipCoordinate, context: Context) -> EthCore | None:
    """Create an Ethernet checker only for architectures with known register maps."""
    if device.is_wormhole():
        return WormholeEthCore(location, context)
    elif device.is_blackhole():
        return BlackholeEthCore(location, context)
    else:
        utils.ERROR(f"Unsupported architecture for check_eth_status: {device._arch}")
        return None


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    BLOCK_TYPES_TO_CHECK = ["active_eth"]
    eth_cores: dict[OnChipCoordinate, EthCore | None] = {}

    def poll_heartbeat(location: OnChipCoordinate) -> None:
        nonlocal has_pending_cores
        if location not in eth_cores:
            eth_cores[location] = get_eth_core(location.device, location, context)
        eth_core = eth_cores[location]
        if eth_core is not None and eth_core.heartbeat_result is None:
            eth_core.poll_heartbeat()
            if eth_core.heartbeat_result is None:
                has_pending_cores = True

    def get_results(location: OnChipCoordinate) -> EthCoreCheckData | None:
        eth_core = eth_cores.get(location)
        # A device read error can prevent a core from completing its heartbeat check.
        if eth_core is not None and eth_core.heartbeat_result is not None:
            return eth_core.get_results()
        return None

    # Sample every pending core before sleeping. Both the wait for a valid sample
    # and the wait for progress overlap across cores. Device reads stay serial.
    # Use RunChecks on every pass to retain its device-skip behavior for read errors.
    while True:
        has_pending_cores = False
        run_checks.run_per_block_check(poll_heartbeat, block_filter=BLOCK_TYPES_TO_CHECK)
        if not has_pending_cores:
            break
        sleep(HEARTBEAT_POLL_INTERVAL_SECONDS)
    return run_checks.run_per_block_check(get_results, block_filter=BLOCK_TYPES_TO_CHECK)


if __name__ == "__main__":
    run_script()
