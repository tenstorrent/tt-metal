import time

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import write_to_device, read_words_from_device

heartbeat_addr = 0x7CC70
metal_fw_exit_flag_addr = 0x3690
host_mailbox_status_addr = 0x7D000
go_msg_addr = 0x690
additional_debug_addr = 0x36B0

context = init_ttexalens()

# Read heartbeat from core e0,4 to e0,11
for col in range(4, 12):
    core = f"e0,{col}"

    metal_fw_flags = read_words_from_device(core, metal_fw_exit_flag_addr, word_count=2)
    host_mailbox_status = read_words_from_device(core, host_mailbox_status_addr, word_count=4)
    go_msg = read_words_from_device(core, go_msg_addr, word_count=1)[0]
    additional_debug = read_words_from_device(core, additional_debug_addr, word_count=3)

    first_read = read_words_from_device(core, heartbeat_addr, word_count=1)[0]
    time.sleep(0.05)
    second_read = read_words_from_device(core, heartbeat_addr, word_count=1)[0]
    heartbeat = first_read != second_read
    print(
        f"Core e0,{col}: metal flag: {metal_fw_flags[0]} {metal_fw_flags[1]}, heartbeat: {heartbeat}, mailbox: {host_mailbox_status[0]:#x} {host_mailbox_status[1]:#x} {host_mailbox_status[2]:#x} {host_mailbox_status[3]:#x}, go_msg: {go_msg:#x}, additional_debug: {additional_debug[0]:#x} {additional_debug[1]:#x} {additional_debug[2]:#x}"
    )
