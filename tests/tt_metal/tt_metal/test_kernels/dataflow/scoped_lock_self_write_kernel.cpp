// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t lock_addr = get_arg(args::lock_addr);
    uint32_t lock_num_elements = get_arg(args::lock_num_elements);
    uint32_t src_buffer_addr = get_arg(args::src_buffer_addr);
    uint32_t write_target_addr = get_arg(args::write_target_addr);
    uint32_t write_size = get_arg(args::write_size);
    uint32_t self_noc_x = get_arg(args::self_noc_x);
    uint32_t self_noc_y = get_arg(args::self_noc_y);

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);
    CoreLocalMem<uint32_t> lock_buffer(lock_addr);

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
