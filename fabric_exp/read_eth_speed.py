#!/usr/bin/env python3
"""Read-only: dump Blackhole ETH firmware boot results (link train status/speed) for every ETH core on every chip.

usage: python_env/bin/python -I fabric_exp/read_eth_speed.py [num_chips]   (run while no other job holds the device)
port_status: 1 UP, 2 DOWN, 3 UNUSED.  train_status: 2 PASS, 0 training, 9/10 timeouts.  train_speed is in Gb/s.

Layout from tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h: boot_results_t at 0x7CC00 begins with
eth_status_t (postcode, port_status, train_status, train_speed, ...; 128 bytes), followed by serdes_results_t
(postcode, serdes_inst, lane_mask, target_speed, data_rate, data_width, ...).
"""
import sys

import tt_umd

BOOT = 0x7CC00
FIELDS = {
    "port_status": BOOT + 0x04,
    "train_status": BOOT + 0x08,
    "train_speed": BOOT + 0x0C,
    "serdes_target_speed": BOOT + 0x80 + 0x0C,
    "serdes_data_rate": BOOT + 0x80 + 0x10,
    "serdes_lane_mask": BOOT + 0x80 + 0x08,
}

num_devices = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for dev_id in range(num_devices):
    dev = tt_umd.TTDevice.create(dev_id)
    dev.init_tt_device()
    soc = dev.get_soc_descriptor()
    for core in soc.get_cores(tt_umd.CoreType.ETH, tt_umd.CoordSystem.NOC0):
        vals = {name: dev.noc_read32(core.x, core.y, addr) for name, addr in FIELDS.items()}
        print(f"chip {dev_id} eth ({core.x:2d},{core.y:2d}): " + " ".join(f"{k}={v}" for k, v in vals.items()))
