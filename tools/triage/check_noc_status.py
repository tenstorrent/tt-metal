#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_noc_status.py

Description:
    This script checks if there are any mismatches between values of number of NOC transactions stored in global
    variables from risc firmware and NOC status registers. These values should be in sync when the NoC is idle.

Owner:
    jbaumanTT
"""

from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.memory_access import MemoryAccess
from ttexalens.tt_exalens_lib import read_register
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from run_checks import run as get_run_checks
from triage import ScriptConfig, log_check_location, run_script

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


def check_noc_status(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    var_to_reg_map: dict[str, str],
    elfs_cache: ElfsCache,
    noc_id: int = 0,
):
    """
    Checks for mismatches between variables and registers that store number of NOC transactions
    and stores them in dictionary creating summary of checking process
    """

    dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)

    fw_elf_path = dispatcher_core_data.firmware_path
    fw_elf = elfs_cache[fw_elf_path]
    kernel_elf = None
    if dispatcher_core_data.kernel_path is not None:
        kernel_elf = elfs_cache[dispatcher_core_data.kernel_path]

    message = f"{risc_name} NOC{noc_id}: "
    passed = True

    loc_mem_access = MemoryAccess.create(location.noc_block.get_risc_debug(risc_name))

    # Skip check when operating in dynamic NOC mode.
    # DM_DEDICATED_NOC is 0 as defined in dev firmware headers (see dev_msgs.h).
    DM_DEDICATED_NOC = 0
    if risc_name == "brisc":
        prev_noc_mode = fw_elf.get_global("prev_noc_mode", loc_mem_access).read_value()
        if prev_noc_mode != DM_DEDICATED_NOC:
            message += "    Skipping NOC status check: prev_noc_mode != DM_DEDICATED_NOC\n"
            log_check_location(location, True, message)
            return

        # Also validate that BRISC's runtime-selected NOC matches the NOC being checked.
        active_noc_index = fw_elf.get_global("noc_index", loc_mem_access).read_value()
        if active_noc_index != noc_id:
            return
    elif kernel_elf is not None:
        # On erisc, the firmware doesn't necessarily select a NOC, so we need to check the kernel ELF.
        noc_mode = kernel_elf.get_global("noc_mode", loc_mem_access).read_value()
        if noc_mode != DM_DEDICATED_NOC:
            message += "    Skipping NOC status check: noc_mode != DM_DEDICATED_NOC\n"
            log_check_location(location, True, message)
            return
        noc_index = kernel_elf.get_global("noc_index", loc_mem_access).read_value()
        if noc_index != noc_id:
            return

    # Check if variables match with corresponding register
    for var in var_to_reg_map:
        reg = var_to_reg_map[var]
        # If reading fails, write error message and skip to next core
        try:
            reg_val = read_register(location=location, register=reg, noc_id=noc_id)
            var_val = fw_elf.get_global(var, loc_mem_access)[noc_id]
        except Exception as e:
            message += "    " + str(e) + "\n"
            passed = False
            break

        if reg_val != var_val:
            # Store name of register and variable where mismatch occurred along side their values
            message += f"    {reg} {var} {reg_val} {var_val}\n"
            passed = False
    if not passed:
        message = (
            "Mismatched state: \n"
            + message
            + "\nEither the device is not idle and is currently processing transactions, the kernel has incorrectly modified the NOC transaction counters, or the NoC is hung."
        )

    log_check_location(location, passed, message)


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    RISC_CORES_TO_CHECK = ["brisc", "erisc", "erisc0", "erisc1"]
    NOC_IDS = [0, 1]
    # Dictionary of corresponding variables and registers to check
    VAR_TO_REG_MAP = {
        "noc_reads_num_issued": "NIU_MST_RD_RESP_RECEIVED",
        "noc_nonposted_writes_num_issued": "NIU_MST_NONPOSTED_WR_REQ_SENT",
        "noc_nonposted_writes_acked": "NIU_MST_WR_ACK_RECEIVED",
        "noc_nonposted_atomics_acked": "NIU_MST_ATOMIC_RESP_RECEIVED",
        "noc_posted_writes_num_issued": "NIU_MST_POSTED_WR_REQ_SENT",
    }

    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    for noc_id in NOC_IDS:
        run_checks.run_per_core_check(
            lambda location, risc_name, _noc_id=noc_id: check_noc_status(
                location, risc_name, dispatcher_data, VAR_TO_REG_MAP, elfs_cache, _noc_id
            ),
            block_filter=BLOCK_TYPES_TO_CHECK,
            core_filter=RISC_CORES_TO_CHECK,
        )


if __name__ == "__main__":
    run_script()
