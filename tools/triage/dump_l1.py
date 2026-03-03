#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_l1 [--dev=<device_id>]...

Options:
    --dev=<device_id>   Specify the device id. 'all' is also an option  [default: all]

Description:
    Dumps the full L1 memory of every tensix and ethernet core on every chip
    into separate files. Files are written to the current working directory with
    the naming convention:
        <hostname>_d<device_id>_(tensix|eth)_x<X>_y<Y>_l1.txt

Owner:
    snijjar
"""

import os
import socket

from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device
from run_checks import run as get_run_checks
from triage import ScriptConfig, run_script, log_check_location

script_config = ScriptConfig(
    depends=["run_checks"],
)

HOSTNAME = socket.gethostname()

# L1 sizes per architecture / core type (bytes)
L1_SIZES = {
    # (is_wormhole, is_blackhole, is_quasar) -> tensix_l1_size
    "tensix": {
        "wormhole": 1464 * 1024,
        "blackhole": 1536 * 1024,
        "quasar": 4 * 1024 * 1024,
    },
    "eth": {
        "wormhole": 256 * 1024 - 32,
        "blackhole": 512 * 1024,
        "quasar": 512 * 1024,
    },
}

# How many bytes to read per transaction (too large may timeout)
READ_CHUNK_SIZE = 4 * 1024


def get_arch_name(device) -> str:
    if device.is_wormhole():
        return "wormhole"
    elif device.is_blackhole():
        return "blackhole"
    elif device.is_quasar():
        return "quasar"
    else:
        raise RuntimeError(f"Unknown architecture for device {device.id}")


def get_core_type_and_l1_size(location: OnChipCoordinate) -> tuple[str, int]:
    device = location._device
    arch = get_arch_name(device)
    block_type = location.noc_block.block_type

    if block_type == "functional_workers":
        return "tensix", L1_SIZES["tensix"][arch]
    else:
        # idle_eth or active_eth
        return "eth", L1_SIZES["eth"][arch]


def dump_core_l1(location: OnChipCoordinate):
    core_type, l1_size = get_core_type_and_l1_size(location)
    device_id = location.device_id

    # Get logical coordinates for the filename
    logical = location.to("logical")
    # logical returns ((x, y), core_type_str) tuple
    coords = logical[0]
    x, y = coords[0], coords[1]

    filename = f"{HOSTNAME}_d{device_id}_{core_type}_x{x}_y{y}_l1.txt"

    log_check_location(location, True, f"Dumping {l1_size} bytes of L1 to {filename}")

    try:
        with open(filename, "w") as f:
            addr = 0
            while addr < l1_size:
                chunk_size = min(READ_CHUNK_SIZE, l1_size - addr)
                data = read_from_device(location, addr=addr, num_bytes=chunk_size, safe_mode=False)
                # Write as hex dump: address followed by hex words
                for offset in range(0, len(data), 16):
                    line_bytes = data[offset : offset + 16]
                    hex_words = " ".join(
                        f"{int.from_bytes(line_bytes[i:i+4], 'little'):08x}"
                        for i in range(0, len(line_bytes), 4)
                        if i + 4 <= len(line_bytes)
                    )
                    f.write(f"{addr + offset:08x}: {hex_words}\n")
                addr += chunk_size
    except Exception as e:
        log_check_location(location, False, f"Failed to dump L1: {e}")
        return None

    log_check_location(location, True, f"Done: {filename}")


def run(args, context: Context):
    run_checks = get_run_checks(args, context)

    BLOCK_TYPES = ["tensix", "idle_eth", "active_eth"]

    run_checks.run_per_block_check(
        lambda location: dump_core_l1(location),
        block_filter=BLOCK_TYPES,
    )


if __name__ == "__main__":
    run_script()
