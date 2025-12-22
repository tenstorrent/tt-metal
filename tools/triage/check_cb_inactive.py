#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_cb_inactive

Description:
    Checks that no command buffers (CB0..CB3) are currently active by reading
    NOC_CMD_CTRL_CB0..3 on both NoCs for tensix and eth blocks. If any is nonzero,
    it prints an error with the device, location, NoC, and CB index.
"""

from ttexalens.coordinate import OnChipCoordinate
from run_checks import run as get_run_checks
from ttexalens.context import Context
from triage import ScriptConfig, log_check_location, run_script

script_config = ScriptConfig(
    depends=["run_checks"],
)


def generate_cb_reg_name(cb_index: int) -> str:
    if not (0 <= cb_index <= 3):
        raise ValueError(f"CB index {cb_index} out of range [0, 3]")

    return f"NOC_CMD_CTRL_CB{cb_index}"


def check_cb_idle(location: OnChipCoordinate, noc_id: int):
    noc_str = f"noc{noc_id}"
    noc_block = location._device.get_block(location)
    register_store = noc_block.get_register_store(noc_id)
    # Read all CB command control registers and emit errors immediately
    for cb_index in range(4):  # CB indices 0-3
        reg_name = generate_cb_reg_name(cb_index)
        value = register_store.read_register(reg_name)
        log_check_location(
            location,
            value == 0,
            f"{noc_str} CB{cb_index} active (0x{value:08X}). NoC is likely hung.",
        )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "eth"]
    run_checks = get_run_checks(args, context)
    run_checks.run_per_block_check(
        lambda location: check_cb_idle(location, noc_id=0), block_filter=BLOCK_TYPES_TO_CHECK
    )
    run_checks.run_per_block_check(
        lambda location: check_cb_idle(location, noc_id=1), block_filter=BLOCK_TYPES_TO_CHECK
    )


if __name__ == "__main__":
    run_script()
