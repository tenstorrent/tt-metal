// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The weight stream of the C++ down projection (see tt/cpp_down.py). The weight is width-sharded
// over the DRAM banks with a core's PN columns contiguous inside one bank's shard row, so each K row
// is ONE read of CHUNK bytes straight from that bank.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t col_off = get_arg_val<uint32_t>(2);

    constexpr uint32_t KB = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(3);
    constexpr uint32_t TILES = get_compile_time_arg_val(4);

    constexpr uint32_t cb = 1;
    uint32_t src = w_addr + col_off;
    for (uint32_t b = 0; b < NB; ++b) {
        cb_reserve_back(cb, KB * TILES);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t k = 0; k < KB; ++k) {
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), l1, CHUNK);
            l1 += CHUNK;
            src += ROW_BYTES;
        }
        noc_async_read_barrier();
        cb_push_back(cb, KB * TILES);
    }
}
