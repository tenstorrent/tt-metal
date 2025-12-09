#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_binary_integrity

Description:
    Checking that code binaries are the way they were uploaded to the device (both firmware and kernel).
"""

from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from run_checks import run as get_run_checks
import os
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device
from triage import ScriptConfig, log_check_risc, run_script

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


def check_binary_integrity(
    location: OnChipCoordinate, risc_name: str, dispatcher_data: DispatcherData, elfs_cache: ElfsCache
):
    dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)

    # Check firmware ELF binary state on the device
    log_check_risc(
        risc_name,
        location,
        os.path.exists(dispatcher_core_data.firmware_path),
        f"Firmware ELF file {dispatcher_core_data.firmware_path} does not exist.",
    )
    if os.path.exists(dispatcher_core_data.firmware_path):
        elf_file = elfs_cache[dispatcher_core_data.firmware_path].elf
        sections_to_verify = [".text"]
        for section_name in sections_to_verify:
            section = elf_file.get_section_by_name(section_name)
            if section is None:
                log_check_risc(
                    risc_name,
                    location,
                    False,
                    f"Section {section_name} not found in ELF file {dispatcher_core_data.firmware_path}.",
                )
            else:
                address: int = section["sh_addr"]
                data: bytes = section.data()
                read_data = read_from_device(location, address, num_bytes=len(data))
                log_check_risc(
                    risc_name,
                    location,
                    read_data == data,
                    f"Data mismatch in section {section_name} at address 0x{address:08x} in ELF file {dispatcher_core_data.firmware_path}.",
                )

    # Check kernel ELF binary state on the device
    if dispatcher_core_data.kernel_xip_path is not None:
        log_check_risc(
            risc_name,
            location,
            os.path.exists(dispatcher_core_data.kernel_xip_path),
            f"Kernel ELF file {dispatcher_core_data.kernel_xip_path} does not exist.",
        )

        if os.path.exists(dispatcher_core_data.kernel_xip_path):
            elf_file = elfs_cache[dispatcher_core_data.kernel_xip_path].elf
            sections_to_verify = [".text"]
            for section_name in sections_to_verify:
                section = elf_file.get_section_by_name(section_name)
                if section is None:
                    log_check_risc(
                        risc_name,
                        location,
                        False,
                        f"Section {section_name} not found in ELF file {dispatcher_core_data.kernel_xip_path}.",
                    )
                else:
                    data: bytes = section.data()
                    address: int = dispatcher_core_data.kernel_offset
                    read_data = read_from_device(location, address, num_bytes=len(data))
                    log_check_risc(
                        risc_name,
                        location,
                        read_data == data,
                        f"Data mismatch in section {section_name} at address 0x{address:08x} in ELF file {dispatcher_core_data.kernel_xip_path}.",
                    )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    run_checks.run_per_core_check(
        lambda location, risc_name: check_binary_integrity(location, risc_name, dispatcher_data, elfs_cache),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
