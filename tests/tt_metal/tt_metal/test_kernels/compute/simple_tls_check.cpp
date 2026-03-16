// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar compute kernel: writes same TLS-check fields as dataflow simple_tls_check
// for parity with GlobalsAndTLS DM test. One slot per compute processor (16 total).
// Uses same slot layout as DM test (simple_tls_check_defines.h).

#include "api/compute/common.h"
#include "dev_mem_map.h"
#include "ckernel.h"
#include "../dataflow/simple_tls_check_defines.h"
#include "api/kernel_thread_globals.h"

// uint32_t shared_global = 5;
// uint32_t uninitialized_global;
// thread_local uint32_t thread_local_var;
// thread_local uint32_t uninitialized_thread_local_var;

void kernel_main() {
    const uint32_t l1_base = get_arg_val<uint32_t>(0);
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    const uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    const uint32_t num_threads = get_num_threads();
    const uint32_t my_thread_id = get_my_thread_id();  // engine id 0-3
    // One slot per compute processor: engine_id * 4 + trisc_id (slot 0-15)
    const uint32_t slot = my_thread_id * NUM_TRISC_CORES + trisc_id;
    // Hardware thread index: DM 0-7, compute 8-23 (8 + neo_id*4 + trisc_id)
    const uint32_t hartid = NUM_DM_CORES + neo_id * NUM_TRISC_CORES + trisc_id;

    DPRINT << "my_thread_id: " << my_thread_id << " num_threads: " << num_threads << ENDL();

    volatile uint32_t tt_l1_ptr* result =
        (volatile uint32_t tt_l1_ptr*)(l1_base + MEM_L1_UNCACHED_BASE + slot * TLS_CHECK_RESULT_SLOT_BYTES);
    result[TLS_CHECK_KERNEL_ID] = 1;  // single kernel
    result[TLS_CHECK_NUM_THREADS] = num_threads;
    result[TLS_CHECK_MY_THREAD_ID] = my_thread_id;
    result[TLS_CHECK_HART_ID] = hartid;
}
