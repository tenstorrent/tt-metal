// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t data_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));

    volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)(data_receiver_semaphore_addr);
    noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
    noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);
}
