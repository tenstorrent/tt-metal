#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_noc_locations

Description:
    Checking that we can reach all NOC endpoints through NOC0 and NOC1.
"""

from ttexalens.coordinate import OnChipCoordinate
from run_checks import run as get_run_checks
from ttexalens.context import Context
from triage import ScriptConfig, log_check, run_script

script_config = ScriptConfig(
    depends=["run_checks"],
)


def check_noc_location(location: OnChipCoordinate, noc_id: int):
    noc_str = f"noc{noc_id}"
    noc_block = location._device.get_block(location)
    register_store = noc_block.get_register_store(noc_id)
    data = register_store.read_register("NOC_NODE_ID")
    n_x = data & 0x3F
    n_y = (data >> 6) & 0x3F
    loc_to_noc = location.to(noc_str)
    log_check(
        loc_to_noc == (n_x, n_y),
        f"Device {location._device._id} {location._device.get_block_type(location)} [{location.to_str('logical')}] block at {location.to_str(noc_str)} has wrong NOC location ({n_x}-{n_y})",
    )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "eth"]
    run_checks = get_run_checks(args, context)
    run_checks.run_per_block_check(
        lambda location: check_noc_location(location, noc_id=0), block_filter=BLOCK_TYPES_TO_CHECK
    )
    run_checks.run_per_block_check(
        lambda location: check_noc_location(location, noc_id=1), block_filter=BLOCK_TYPES_TO_CHECK
    )


if __name__ == "__main__":
    run_script()
