#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import sys
import struct

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device

context: Context = init_ttexalens()

HEARTBEAT_ADDR = 0x1c
READ_COUNT = 8
READ_SIZE = 4  # uint32

not_beating = []

for device_id in range(8):  # Assuming up to 8 devices
    device = context.devices[device_id]
    n_eth_chans = 16
    
    for ch in range(n_eth_chans):
        # loc = OnChipCoordinate(device_id, 0, 0, ch)
        loc = OnChipCoordinate(0, ch, "logical", device, "eth")
        values = []
        for _ in range(READ_COUNT):
            # Raw read from address 0x1c
            read_data = read_from_device(loc, HEARTBEAT_ADDR, device_id, READ_SIZE, context)
            val = struct.unpack('<I', bytes(read_data))[0]
            values.append(val)
            time.sleep(0.0001)

        # Check if values are changing
        beating = len(set(values)) > 1  # If not all same, it's changing

        coord_str = loc.to_user_str()
        print(f"Device {device_id} Ethernet core at D {device_id} (E{coord_str}, {HEARTBEAT_ADDR}): {'Beating' if beating else 'Not beating'}")
        print(f"  Values: {values}")
        if not beating:
            not_beating.append({"chip": device_id, "coord": coord_str})

# pretty print this as a table
print(f"Number of non-responsive cores: {len(not_beating)}")
print(f"{'Chip':<5} {'Coord':<10} {'Beating':<10}")
for item in not_beating:
    print(f"{item['chip']:<5} {item['coord']:<10} {'No'}")