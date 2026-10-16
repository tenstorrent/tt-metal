// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): the chips' copies of this core's compact tiles from the row all-gather of DownStream's compact
// [1, 1, 32, hidden / 8] output ([1, 1, 32, chips * hidden / 8]: chip c's compact tile tc is page c * compact_tiles +
// tc), one tile's chips copies per push, for moe_reduce_compute.cpp to sum.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t chips = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);

    const InterleavedAddrGen<true> in_gen = {.bank_base_address = in_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < tiles; i++) {
        cb_reserve_back(cb_in, chips);
        const uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t c = 0; c < chips; c++) {
            noc_async_read(in_gen.get_noc_addr(c * compact_tiles + first + i, 0), l1 + c * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, chips);
    }
}
