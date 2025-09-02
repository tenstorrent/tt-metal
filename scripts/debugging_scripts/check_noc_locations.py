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
from check_per_block_location import run as get_check_per_block_location, PerBlockLocationCheck
from ttexalens.context import Context
from block_locations_to_check import BlockType
from triage import ScriptConfig, log_check, run_script

script_config = ScriptConfig(
    depends=["block_locations_to_check"],
)


def check_noc_location(location: OnChipCoordinate, block_type: BlockType, noc_id: int):
    noc_str = f"noc{noc_id}"
    noc_block = location._device.get_block(location)
    register_store = noc_block.get_register_store(noc_id)
    data = register_store.read_register("NOC_NODE_ID")
    n_x = data & 0x3F
    n_y = (data >> 6) & 0x3F
    loc_to_noc = location.to(noc_str)
    log_check(
        loc_to_noc == (n_x, n_y),
        f"Device {location._device._id} {block_type} [{location.to_str('logical')}] block at {location.to_str(noc_str)} has wrong NOC location ({n_x}-{n_y})",
    )


def run(args, context: Context):
    block_locations_to_check: PerBlockLocationCheck = get_check_per_block_location(args, context)
    block_locations_to_check.run_check(
        lambda location, block_type: check_noc_location(location, block_type, noc_id=0),
    )

    block_locations_to_check.run_check(
        lambda location, block_type: check_noc_location(location, block_type, noc_id=1),
    )


if __name__ == "__main__":
    run_script()
