#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    scripts/debugging_scripts/check_eth_status.py

Description:
    Check status on the ethernet cores
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from triage import ScriptConfig, triage_field, log_check, run_script
from ttexalens.context import Context
from ttexalens.device import Device, OnChipCoordinate
from ttexalens.register_store import read_word_from_device

from check_per_device import run as get_check_per_device

from ttexalens.hw.tensix.blackhole.blackhole import BlackholeDevice
from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice

script_config = ScriptConfig(
    depends=["check_per_device"],
)


def log_check_with_loc(loc: OnChipCoordinate, ok: bool, message: str):
    log_check(ok, f"{loc._device._id} {loc.to_user_str()}: {message}")


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
        log_check_with_loc(self.location, False, f"No heartbeat detected for {self.location.to_user_str()}")
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
            log_check_with_loc(self.location, port_status_str != "Down", f"{self.location.to_user_str()} port is down")

        # RETRAIN COUNT
        output.retrain_count = int(
            read_word_from_device(self.location, self.eth_core_definitions.retrain_count, context=self.context)
        )
        log_check_with_loc(
            self.location,
            not output.retrain_count,
            f"{self.location.to_user_str()} retrain count is {output.retrain_count}",
        )

        # RX LINK UP
        output.rx_link_up = (
            "Up"
            if read_word_from_device(self.location, self.eth_core_definitions.rx_link_up, context=self.context)
            else "Down"
        )
        log_check_with_loc(
            self.location,
            not any_pending_message,
            f"{self.location.to_user_str()} mailbox: {output.mailbox} (pending message)",
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
                log_check_with_loc(
                    self.location,
                    not any_pending_message,
                    f"{self.location.to_user_str()} mailbox: {output.mailbox} (pending message)",
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
    if type(device) == WormholeDevice:
        eth_core = WormholeEthCore(location, context)
        eth_core.get_results()
    elif type(device) == BlackholeDevice:
        eth_core = BlackholeEthCore(location, context)
        eth_core.get_results()
    else:
        utils.ERROR(f"Unsupported architecture for check_eth_status: {device._arch}")


def run_checks(device: Device, context: Context):
    # Loop through all active ethernet cores
    locations = device.get_block_locations(block_type="eth")
    for loc in locations:
        noc_block = device.get_block(loc)
        if noc_block not in device.active_eth_blocks:
            continue
        get_eth_core_data(device, loc, context)


def run(args, context: Context):
    check_per_device = get_check_per_device(args, context)
    return check_per_device.run_check(lambda device: run_checks(device, context))


if __name__ == "__main__":
    run_script()
