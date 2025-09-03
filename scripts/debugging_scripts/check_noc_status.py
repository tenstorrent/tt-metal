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
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.parse_elf import ELFFile, mem_access
from ttexalens.firmware import ELF

from dispatcher_data import run as get_dispatcher_data, DispatcherData
from check_per_block_location import run as get_check_per_block_location
from triage import ScriptConfig, log_check, run_script

script_config = ScriptConfig(
    depends=["check_per_block_location", "dispatcher_data"],
)


def check_noc_status(
    location: OnChipCoordinate,
    dispatcher_data: DispatcherData,
    context: Context,
    var_to_reg_map: dict[str, str],
    risc_name: str = "brisc",
    noc_id: int = 0,
):
    """
    Checks for mismatches between variables and registers that store number of NOC transactions
    and stores them in dictionary creating summary of checking process
    """

    # Risc cores of same type have the same firmware so we cache it by risc name
    if not hasattr(check_noc_status, "fw_elf_cache"):
        check_noc_status.fw_elf_cache: dict[str, ELFFile] = {}

    if risc_name not in check_noc_status.fw_elf_cache:
        fw_elf_path = dispatcher_data.get_core_data(location, risc_name).firmware_path
        check_noc_status.fw_elf_cache[risc_name] = parse_elf(fw_elf_path, context)

    fw_elf = check_noc_status.fw_elf_cache[risc_name]

    message = f"Device {location._device._id} at {location.to_user_str()}\n"
    passed = True

    loc_mem_reader = ELF.get_mem_reader(location, risc_name)

    # Check if variables match with corresponding register
    for var in var_to_reg_map:
        reg = var_to_reg_map[var]
        # If reading fails, write error message and skip to next core
        try:
            reg_val = read_tensix_register(location=location, register=reg, noc_id=noc_id)
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
    BLOCK_TYPES_TO_CHECK = "tensix"
    RISC_NAME = "brisc"
    NOC_ID = 0
    # Dictionary of corresponding variables and registers to check
    VAR_TO_REG_MAP = {
        "noc_reads_num_issued": "NIU_MST_RD_RESP_RECEIVED",
        "noc_nonposted_writes_num_issued": "NIU_MST_NONPOSTED_WR_REQ_SENT",
        "noc_nonposted_writes_acked": "NIU_MST_WR_ACK_RECEIVED",
        "noc_nonposted_atomics_acked": "NIU_MST_ATOMIC_RESP_RECEIVED",
        "noc_posted_writes_num_issued": "NIU_MST_POSTED_WR_REQ_SENT",
    }

    dispatcher_data = get_dispatcher_data(args, context)
    check_per_block_location = get_check_per_block_location(args, context)
    check_per_block_location.run_check(
        lambda location: check_noc_status(location, dispatcher_data, context, VAR_TO_REG_MAP, RISC_NAME, NOC_ID),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
