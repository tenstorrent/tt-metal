// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t noc0_num_senders = get_compile_time_arg_val(0);
    constexpr uint32_t noc1_num_senders = get_compile_time_arg_val(1);
    uint32_t noc0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t noc1_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    volatile tt_l1_ptr uint32_t* noc0_receiver_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)noc0_receiver_semaphore_addr;
    volatile tt_l1_ptr uint32_t* noc1_receiver_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)noc1_receiver_semaphore_addr;
    noc_semaphore_wait(noc0_receiver_semaphore_addr_ptr, noc0_num_senders);
    noc_semaphore_wait(noc1_receiver_semaphore_addr_ptr, noc1_num_senders);
}
