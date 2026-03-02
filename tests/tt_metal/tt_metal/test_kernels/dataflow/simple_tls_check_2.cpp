// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "simple_tls_check_defines.h"

uint32_t shared_global_2 = 5;
uint32_t uninitialized_global_2;
thread_local uint32_t thread_local_2;
thread_local uint32_t uninitialized_thread_local_2;

void kernel_main() {
    const uintptr_t signal_address = get_arg_val<uint32_t>(0);
    const uint32_t dram_dst_address = get_arg_val<uint32_t>(1);
    const uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(2);
    const uint32_t base_l1_result_addr = get_arg_val<uint32_t>(3);

    const uint32_t kernel_id = 2;
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));

    uint32_t slot_offset = (uint32_t)hartid * TLS_CHECK_RESULT_SLOT_BYTES;
    uint32_t l1_result_addr = base_l1_result_addr + MEM_L1_UNCACHED_BASE;

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

    uint32_t global_start = shared_global_2;
    shared_global_2 = shared_global_2 + 1;
    uint32_t global_end = shared_global_2;
    uint64_t global_addr = (uint64_t)(&shared_global_2);

    uint32_t uninitialized_global_start = uninitialized_global_2;
    uninitialized_global_2 = uninitialized_global_2 + 1;
    uint32_t uninitialized_global_end = uninitialized_global_2;

    uint32_t thread_local_start = thread_local_2;
    thread_local_2 = thread_local_2 + 1;
    uint32_t thread_local_end = thread_local_2;
    uint64_t thread_local_addr = (uint64_t)(&thread_local_2);

    uint32_t uninitialized_thread_local_start = uninitialized_thread_local_2;
    uninitialized_thread_local_2 = uninitialized_thread_local_2 + 1;
    uint32_t uninitialized_thread_local_end = uninitialized_thread_local_2;

    volatile tt_l1_ptr uint32_t* result = (tt_l1_ptr uint32_t*)((uintptr_t)l1_result_addr);
    result[TLS_CHECK_KERNEL_ID] = kernel_id;
    result[TLS_CHECK_NUM_KERNEL_THREADS] = get_num_kernel_threads();
    result[TLS_CHECK_MY_THREAD_ID] = get_my_thread_id();
    result[TLS_CHECK_HART_ID] = (uint32_t)hartid;
    result[TLS_CHECK_THREAD_0_HART_ID] = thread_0_hartid;
    result[TLS_CHECK_GLOBAL_START] = global_start;
    result[TLS_CHECK_GLOBAL_END] = global_end;
    result[TLS_CHECK_GLOBAL_ADDR_LO] = (uint32_t)(global_addr & 0xFFFFFFFFu);
    result[TLS_CHECK_GLOBAL_ADDR_HI] = (uint32_t)(global_addr >> 32);
    result[TLS_CHECK_UNINITIALIZED_GLOBAL_START] = uninitialized_global_start;
    result[TLS_CHECK_UNINITIALIZED_GLOBAL_END] = uninitialized_global_end;
    result[TLS_CHECK_THREAD_LOCAL_START] = thread_local_start;
    result[TLS_CHECK_THREAD_LOCAL_END] = thread_local_end;
    result[TLS_CHECK_THREAD_LOCAL_ADDR_LO] = (uint32_t)(thread_local_addr & 0xFFFFFFFFu);
    result[TLS_CHECK_THREAD_LOCAL_ADDR_HI] = (uint32_t)(thread_local_addr >> 32);
    result[TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_START] = uninitialized_thread_local_start;
    result[TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_END] = uninitialized_thread_local_end;

    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_dst_bank_id, dram_dst_address + slot_offset);
    noc_async_write(l1_result_addr, dram_noc_addr, TLS_CHECK_RESULT_SLOT_BYTES);
    noc_async_write_barrier();

    *signal_addr = hartid + 1;
}
