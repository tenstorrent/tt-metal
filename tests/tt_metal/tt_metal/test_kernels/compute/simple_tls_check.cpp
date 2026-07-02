// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar compute: parity with dataflow simple_tls_check for globals, TLS, and
// kernel_text_offset grouping. Built for all four TRISC translation units;
// each ELF carries its own statics. Threading metadata matches DM-style
// software threads across NEOs for a given TRISC within each engine.

#include "api/compute/common.h"
#include "ckernel.h"
#include "dev_mem_map.h"
#include "hostdev/dev_msgs.h"
#include "../dataflow/simple_tls_check_defines.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

uint32_t shared_global = 5;
uint32_t uninitialized_global;
thread_local uint32_t thread_local_var = 10;
thread_local uint32_t uninitialized_thread_local_var;

void kernel_main() {
    const uint32_t l1_base = get_arg(args::l1_result_addr);
    const uintptr_t signal_address = get_arg(args::signal_address);
    constexpr uint32_t kernel_id = 1;

    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    const uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    const uint32_t slot = neo_id * NUM_TRISC_CORES + trisc_id;
    const uint32_t hartid = NUM_DM_CORES + slot;

    uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    launch_msg_t tt_l1_ptr* launch_msg = &(*GET_MAILBOX_ADDRESS_DEV(launch))[launch_idx];
    uint32_t my_kt = launch_msg->kernel_config.kernel_text_offset[hartid];
    uint32_t thread_0_hartid = MaxProcessorsPerCoreType;
    for (uint32_t j = NUM_DM_CORES; j < MaxProcessorsPerCoreType; j++) {
        if (launch_msg->kernel_config.kernel_text_offset[j] == my_kt) {
            thread_0_hartid = j;
            break;
        }
    }

    volatile tt_l1_ptr uint32_t* signal_addr = (volatile tt_l1_ptr uint32_t*)((uintptr_t)signal_address);
    while (*signal_addr != hartid) {
    }

    uint32_t global_start = shared_global;
    shared_global = shared_global + 1;
    uint32_t global_end = shared_global;
    uint64_t global_addr = (uint64_t)(&shared_global);

    uint32_t uninitialized_global_start = uninitialized_global;
    uninitialized_global = uninitialized_global + 1;
    uint32_t uninitialized_global_end = uninitialized_global;

    uint32_t thread_local_start = thread_local_var;
    thread_local_var = thread_local_var + 1;
    uint32_t thread_local_end = thread_local_var;
    uint64_t thread_local_addr = (uint64_t)(&thread_local_var);

    uint32_t uninitialized_thread_local_start = uninitialized_thread_local_var;
    uninitialized_thread_local_var = uninitialized_thread_local_var + 1;
    uint32_t uninitialized_thread_local_end = uninitialized_thread_local_var;

    volatile uint32_t tt_l1_ptr* result =
        (volatile uint32_t tt_l1_ptr*)(l1_base + MEM_L1_UNCACHED_BASE + slot * TLS_CHECK_RESULT_SLOT_BYTES);
    result[TLS_CHECK_KERNEL_ID] = kernel_id;
    result[TLS_CHECK_NUM_THREADS] = get_num_threads();
    result[TLS_CHECK_MY_THREAD_ID] = get_my_thread_id();
    result[TLS_CHECK_HART_ID] = hartid;
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

    *signal_addr = hartid + 1;
}
