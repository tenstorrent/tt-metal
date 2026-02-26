// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"


uint32_t shared_global_3 = 5;
//thread_local uint32_t thread_local_global = 10;

void kernel_main() {
    const uintptr_t signal_address = get_arg_val<uint32_t>(0);
    const uint32_t dram_dst_address = get_arg_val<uint32_t>(1);

    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));

    // Obtain launch message from mailbox and derive thread 0 (lowest hartid with same kernel).
    uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    launch_msg_t tt_l1_ptr* launch_msg = &(*GET_MAILBOX_ADDRESS_DEV(launch))[launch_idx];
    uint32_t my_kt = launch_msg->kernel_config.kernel_text_offset[hartid];
    uint32_t thread_0_hartid = MaxDMProcessorsPerCoreType;
    for (uint32_t j = 0; j < MaxDMProcessorsPerCoreType; j++) {
        if (launch_msg->kernel_config.kernel_text_offset[j] == my_kt) {
            thread_0_hartid = j;
            break;
        }
    }

    volatile tt_l1_ptr std::uint32_t* signal_addr = (tt_l1_ptr uint32_t*)((uintptr_t)signal_address);
    while (*signal_addr != hartid);

    DPRINT << "KERNEL 3 START" << ENDL();
    DPRINT << "num_kernel_threads: " << get_num_kernel_threads() << ENDL();
    DPRINT << "my_thread_id: " << get_my_thread_id() << ENDL();

    DPRINT << "hartid: " << hartid << ", thread_0_hartid: " << thread_0_hartid << ENDL();

    DPRINT << "start global value: " << shared_global_3 << ENDL();
//    DPRINT << "start thread local value: " << thread_local_global << ENDL();
    shared_global_3 = shared_global_3 + 1;
//    thread_local_global = thread_local_global + 1;
    DPRINT << "end global value: " << shared_global_3 << ENDL();
    DPRINT << "global address: " << (uint64_t)(&shared_global_3) << ENDL();
//    DPRINT << "end thread local value: " << thread_local_global << ENDL();

    *signal_addr = hartid + 1;

//    *((uint32_t*)(dst_addr + MEM_L1_UNCACHED_BASE)) =
//        value;  // use cache write-around for now, in the future use cache flush
}
