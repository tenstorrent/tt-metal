#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
from run_checks import run as get_run_checks
from triage import ScriptConfig, triage_field, log_check_location, run_script
from ttexalens import read_word_from_device
from ttexalens.context import Context
from ttexalens.device import Device, OnChipCoordinate
import utils

script_config = ScriptConfig(
    depends=["run_checks"],
)


@dataclass
class EthCoreDefinitions:
    """Arch specific addresses of various ethernet core fields"""

    port_status: int
    retrain_count: int
    rx_link_up: int
    heartbeat: int
    mailbox: int
    mailbox_slots: int


@dataclass
class EthCoreCheckData:
    location: OnChipCoordinate = triage_field("Loc")
    port_status: str = triage_field("Port Status")
    retrain_count: int = triage_field("Retrain Count")
    rx_link_up: str = triage_field("RX Link Up")
    heartbeat: bool = triage_field("Heartbeat")
    mailbox: list[int] = triage_field("Mailbox")

    def __init__(self):
        self.location = None
        self.port_status = None
        self.retrain_count = None
        self.rx_link_up = None
        self.heartbeat = None
        self.mailbox = None


class EthCore(ABC):
    """
    Base class for Ethernet cores that provides common functionality.
    """

    eth_core_definitions: EthCoreDefinitions

    def __init__(self, location: OnChipCoordinate, context: Context):
        self.location = location
        self.context = context

    @abstractmethod
    def port_status_to_string(self, port_status: int) -> str:
        """Convert port status value to a readable string."""
        pass

    def check_for_heartbeat(self):
        """Check for heartbeat at the heartbeat address."""
        previous_data = 0
        # Check for a changing value at the heartbeat address. Read up to 100 times
        for i in range(100):
            read_data = read_word_from_device(self.location, self.eth_core_definitions.heartbeat, context=self.context)
            if read_data != previous_data:
                return True
            previous_data = read_data
        log_check_location(self.location, False, "No heartbeat detected")
        return False

    def get_results(self):
        """Get and log all ethernet core status results."""
        output = EthCoreCheckData()
        # HEARTBEAT
        output.heartbeat = self.check_for_heartbeat()

        # PORT STATUS
        if self.eth_core_definitions.port_status is not None:
            port_status = read_word_from_device(
                self.location, self.eth_core_definitions.port_status, context=self.context
            )
            port_status_str = self.port_status_to_string(port_status)
            if port_status_str == None:
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

    def port_status_to_string(self, port_status: int) -> str:
        """Convert Wormhole port status to readable string."""
        # Undefined right now for Wormhole. Need to find the mapping
        status_map = {0: "Undefined", 1: "Undefined", 2: "Undefined", 3: "Undefined"}
        return status_map.get(port_status, None)


class BlackholeEthCore(EthCore):
    """Blackhole-specific Ethernet core implementation."""

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

    def port_status_to_string(self, port_status: int) -> str:
        """Convert Blackhole port status to readable string."""
        status_map = {0: None, 1: "Up", 2: "Down", 3: "Unused"}
        return status_map.get(port_status, None)


def get_eth_core_data(device: Device, location: OnChipCoordinate, context: Context) -> EthCoreCheckData:
    """Create appropriate EthCore instance based on device type and get results."""
    if device.is_wormhole():
        eth_core = WormholeEthCore(location, context)
        eth_core.get_results()
    elif device.is_blackhole():
        eth_core = BlackholeEthCore(location, context)
        eth_core.get_results()
    else:
        utils.ERROR(f"Unsupported architecture for check_eth_status: {device._arch}")


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    BLOCK_TYPES_TO_CHECK = ["active_eth"]
    run_checks.run_per_block_check(
        lambda location: get_eth_core_data(location._device, location, context), block_filter=BLOCK_TYPES_TO_CHECK
    )


if __name__ == "__main__":
    run_script()
