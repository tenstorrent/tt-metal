// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t lock_addr = get_arg_val<uint32_t>(0);
    uint32_t lock_num_elements = get_arg_val<uint32_t>(1);
    uint32_t src_buffer_addr = get_arg_val<uint32_t>(2);
    uint32_t write_target_addr = get_arg_val<uint32_t>(3);
    uint32_t write_size = get_arg_val<uint32_t>(4);
    uint32_t self_noc_x = get_arg_val<uint32_t>(5);
    uint32_t self_noc_y = get_arg_val<uint32_t>(6);

    experimental::Noc noc;
    experimental::UnicastEndpoint unicast_endpoint;
    experimental::CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);
    experimental::CoreLocalMem<uint32_t> lock_buffer(lock_addr);

    {
        auto lock = lock_buffer.scoped_lock(lock_num_elements);
        noc.async_write(
            src_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = self_noc_x, .noc_y = self_noc_y, .addr = write_target_addr});
        noc.async_write_barrier();
    }
}
