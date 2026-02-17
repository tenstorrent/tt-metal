// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    riscv_wait(START_DELAY);

    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> local_buffer(L1_BUFFER_ADDR);
    experimental::UnicastEndpoint unicast_endpoint;

    constexpr uint32_t num_bytes = 64;
    for (uint32_t i = 0; i < 10000; ++i) {
        noc.async_read(
            unicast_endpoint,
            local_buffer,
            num_bytes,
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = i,
            },
            {});
        noc.async_read_barrier();
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            num_bytes,
            {},
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = i,
            });
        noc.async_write_barrier();
    }
}
