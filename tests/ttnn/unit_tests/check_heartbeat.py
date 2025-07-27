# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import os

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import write_to_device, read_words_from_device

heartbeat_addr = 0x7CC70
metal_fw_exit_flag_addr = 0x3690
host_mailbox_status_addr = 0x7D000
go_msg_addr = 0x690
additional_debug_addr = 0x36B0
run_count_addr = 0x3680
port_status_addr = 0x7CC04

context = init_ttexalens()

# Output all the erisc count files
# for f in erisc_count*; do echo "--- $f ---"; cat "$f"; echo; done


def print_erisc_count_files():
    btu_files = [f for f in os.listdir(".") if f.startswith("erisc_count_")]
    for fname in sorted(btu_files):
        print(f"--- {fname} ---")
        try:
            with open(fname, "r") as f:
                print(f.read())
        except Exception as e:
            print(f"Could not read {fname}: {e}")
        print()


print_erisc_count_files()

# Read heartbeat from core e0,4 to e0,11
for col in range(4, 12):
    core = f"e0,{col}"

    metal_fw_flags = read_words_from_device(core, metal_fw_exit_flag_addr, word_count=2)
    host_mailbox_status = read_words_from_device(core, host_mailbox_status_addr, word_count=4)
    go_msg = read_words_from_device(core, go_msg_addr, word_count=1)[0]
    additional_debug = read_words_from_device(core, additional_debug_addr, word_count=3)
    run_count = read_words_from_device(core, run_count_addr, word_count=1)[0]
    port_status = read_words_from_device(core, port_status_addr, word_count=1)[0]

    first_read = read_words_from_device(core, heartbeat_addr, word_count=1)[0]
    time.sleep(0.05)
    second_read = read_words_from_device(core, heartbeat_addr, word_count=1)[0]
    heartbeat = first_read != second_read
    print(
        f"Core e0,{col}: port: {port_status}, run count: {run_count}, metal flag: {metal_fw_flags[0]} {metal_fw_flags[1]}, heartbeat: {heartbeat}, mailbox: {host_mailbox_status[0]:#x} {host_mailbox_status[1]:#x} {host_mailbox_status[2]:#x} {host_mailbox_status[3]:#x}, go_msg: {go_msg:#x}, additional_debug: {additional_debug[0]:#x} {additional_debug[1]:#x} {additional_debug[2]:#x}"
    )

raise Exception("Running this means the test failed")
