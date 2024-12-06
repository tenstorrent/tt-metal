// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr std::uint32_t iteration = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);

    std::uint32_t noc_x = get_arg_val<uint32_t>(0);
    std::uint32_t noc_y = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;
    uint32_t l1_read_addr = get_read_ptr(cb_id);

    DPRINT << "Start" <<ENDL();

    for (uint32_t i = 0; i < iteration; i ++) {
        uint32_t noc = (i % 2) == 0 ? noc_index : 1-noc_index;
        uint64_t noc_addr = get_noc_addr(noc_x, noc_y, l1_read_addr, noc);
        noc_async_read(noc_addr, l1_read_addr, page_size, noc);
        noc_semaphore_inc(noc_addr, 1, noc);
        noc_async_write(l1_read_addr, noc_addr, page_size, noc);
        noc_async_write_one_packet(l1_read_addr, noc_addr, page_size, noc);
    }
    DPRINT << "END" <<ENDL();

    noc_async_write_barrier(noc_index);
    noc_async_write_barrier(1-noc_index);

}
