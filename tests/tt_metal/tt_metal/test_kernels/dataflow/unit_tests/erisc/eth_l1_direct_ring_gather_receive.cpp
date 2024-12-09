// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint32_t num_bytes = get_arg_val<uint32_t>(0);
    std::uint32_t num_transfers = get_arg_val<uint32_t>(1);
    std::uint32_t sem_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(1);

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sem_addr);

    for (uint32_t i = 0; i < num_transfers; ++i) {
        eth_wait_for_bytes(num_bytes);
        noc_semaphore_inc(sender_semaphore_noc_addr, 1);
        eth_noc_semaphore_wait(receiver_semaphore_addr_ptr, 1);
        noc_semaphore_set(receiver_semaphore_addr_ptr, 0);
        eth_receiver_done();
    }
}
