#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    scripts/debugging_scripts/check_noc_status.py <elf-file> [-v]

Arguments:
    <elf-file>  Path to risc firmware elf file

Options:
    -v  If true includes passed tests in optput. Default: False

Description:
    This script checks if there are any mismatches between values of number of NOC transactions
    stored in global variables from risc firmware and NOC status registers. Script looks for
    these mismatches across all available devices and locations.
"""

import sys
import os

try:
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.tt_exalens_lib import read_riscv_memory, read_tensix_register
    from ttexalens import util
    from ttexalens.parse_elf import decode_symbols
    from ttexalens.context import Context
    from ttexalens.tt_exalens_lib import parse_elf
    from ttexalens.parse_elf import mem_access
    from ttexalens.firmware import ELF
except Exception as e:
    print("No tt-exalens detected. Please install tt-exalens with:\n ./scripts/install_debugger.sh")
    sys.exit(1)

from elftools.elf.elffile import ELFFile
from docopt import docopt


def check_noc_status(fw_elf, context: Context, risc_name: str = "brisc", noc_id: int = 0) -> dict:
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

    summary = {}
    # Loop through all available devices
    for device_id in context.device_ids:
        device = context.devices[device_id]
        # Get all functional workers and loop through them
        locations = device.get_block_locations(block_type="functional_workers")
        for loc in locations:
            passed = True
            error = False

            loc_mem_reader = ELF.get_mem_reader(loc, risc_name)

            # Check if variables match with corresponding register
            for var in VAR_TO_REG_MAP:
                reg = VAR_TO_REG_MAP[var]
                # If reading fails, write error message and skip to next core
                try:
                    reg_val = read_tensix_register(core_loc=loc, register=reg, noc_id=noc_id, context=context)
                    var_val = mem_access(fw_elf, var, loc_mem_reader)[0][0]
                except Exception as e:
                    summary[(device_id, loc)] = str(e)
                    error = True
                    break

                if reg_val != var_val:
                    # Store name of register and variable where mismatch occured along side their values
                    if passed:
                        # If this is the first one to fail, init list
                        summary[(device_id, loc)] = [[reg, var, reg_val, var_val]]
                    else:
                        summary[(device_id, loc)].append([reg, var, reg_val, var_val])
                    passed = False

            # If core passed the inspection, write passed
            if passed and not error:
                summary[(device_id, loc)] = f"PASSED"

    return summary


def print_summary(summary: dict, verbose: bool = False) -> None:
    """Prints summary of checking NOC transactions status"""
    all_passed = True
    for key in summary.keys():
        if not verbose and summary[key] == "PASSED":
            continue

        device_id, loc = key
        util.INFO(f"Device: {device_id}, loc: {loc}", end=" ")
        if isinstance(summary[key], str):
            if summary[key] == "PASSED":
                print(f"{util.CLR_GREEN}{summary[key]}{util.CLR_END}")
            else:
                all_passed = False
                util.WARN(summary[key])
        else:
            all_passed = False
            util.ERROR("FAILED")
            for elem in summary[key]:
                reg, var, reg_val, var_val = elem
                util.ERROR(f"\tMismatch between {reg} and {var} -> {reg_val} != {var_val}")

    if all_passed:
        print(f"\n{util.CLR_GREEN}All tests passed!{util.CLR_END}")


def main():
    args = docopt(__doc__, argv=sys.argv[1:])
    elf_path = args["<elf-file>"]
    verbose = True if args["-v"] else False

    if not os.path.exists(elf_path):
        util.ERROR(f"File {elf_path} does not exist")
        return

    context = init_ttexalens()
    # # Get symbols in order to obtain variable addresses
    # symbols = get_symbols_from_elf(elf_path, context)
    fw_elf = parse_elf(elf_path)

    risc_name = "brisc"  # For now only works on BRISC
    noc_id = 0  # For now we only use noc0

    summary = check_noc_status(fw_elf, context, risc_name, noc_id)
    print_summary(summary, verbose)


if __name__ == "__main__":
    main()
