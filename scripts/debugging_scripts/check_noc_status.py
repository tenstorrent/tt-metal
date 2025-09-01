#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    scripts/debugging_scripts/check_noc_status.py

Description:
    This script checks if there are any mismatches between values of number of NOC transactions
    stored in global variables from risc firmware and NOC status registers.
"""

from ttexalens.tt_exalens_lib import read_tensix_register, parse_elf
from ttexalens.context import Context
from ttexalens.device import Device, OnChipCoordinate
from ttexalens.parse_elf import mem_access
from ttexalens.firmware import ELF

from check_per_device import run as get_check_per_device
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from block_locations_to_check import run as get_block_locations_to_check
from triage import ScriptConfig, log_check, run_script

script_config = ScriptConfig(
    depends=["check_per_device", "dispatcher_data", "block_locations_to_check"],
)


def check_noc_status(
    device: Device,
    dispatcher_data: DispatcherData,
    context: Context,
    locations: list[OnChipCoordinate],
    risc_name: str = "brisc",
    noc_id: int = 0,
):
    """
    Checks for mismatches between variables and registers that store number of NOC transactions
    and stores them in dictionary creating summary of checking process
    """

    # Dictionary of corresponding variables and registers to check
    VAR_TO_REG_MAP = {
        "noc_reads_num_issued": "NIU_MST_RD_RESP_RECEIVED",
        "noc_nonposted_writes_num_issued": "NIU_MST_NONPOSTED_WR_REQ_SENT",
        "noc_nonposted_writes_acked": "NIU_MST_WR_ACK_RECEIVED",
        "noc_nonposted_atomics_acked": "NIU_MST_ATOMIC_RESP_RECEIVED",
        "noc_posted_writes_num_issued": "NIU_MST_POSTED_WR_REQ_SENT",
    }

    # Since all firmware elfs are the same, we can query dispatcher data and parse elf only once
    fw_elf_path = dispatcher_data.get_core_data(locations[0], risc_name).firmware_path
    fw_elf = parse_elf(fw_elf_path, context)

    for loc in locations:
        message = f"Device {device._id} at {loc.to_user_str()}\n"
        passed = True

        loc_mem_reader = ELF.get_mem_reader(loc, risc_name)

        # Check if variables match with corresponding register
        for var in VAR_TO_REG_MAP:
            reg = VAR_TO_REG_MAP[var]
            # If reading fails, write error message and skip to next core
            try:
                reg_val = read_tensix_register(location=loc, register=reg, noc_id=noc_id, context=context)
                var_val = mem_access(fw_elf, var, loc_mem_reader)[0][0]
            except Exception as e:
                message += "    " + str(e) + "\n"
                passed = False
                break

            if reg_val != var_val:
                # Store name of register and variable where mismatch occurred along side their values
                message += f"    {reg} {var} {reg_val} {var_val}\n"
                passed = False

        log_check(passed, message)


def run(args, context: Context):
    check_per_device = get_check_per_device(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    block_locations_to_check = get_block_locations_to_check(args, context)
    check_per_device.run_check(
        lambda device: check_noc_status(
            device, dispatcher_data, context, block_locations_to_check[device]["tensix"], risc_name="brisc", noc_id=0
        )
    )


if __name__ == "__main__":
    run_script()
