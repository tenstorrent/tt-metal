// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Big-M streamed-expert x reader (broadcaster NCRISC): streams x from an interleaved DRAM tensor, one K-block
// (X_BLK_TILES consecutive tiles, [MT x KBLK] row-major) at a time, into the staging CB the broadcaster multicasts
// from.
//
// CT: 0 CB, 1 X_BLK_TILES, 2 TILE_BYTES, 3 NUM_BLOCKS
// RT: 0 x DRAM address
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define SE_MARK(name)            \
    {                            \
        DeviceZoneScopedN(name); \
    }
#else
#define SE_MARK(name)
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    const InterleavedAddrGenFast<true> x = {
        .bank_base_address = get_arg_val<uint32_t>(0), .page_size = tile_bytes, .data_format = DataFormat::Bfp8_b};
    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_reserve_back(cb, blk);
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t t = 0; t < blk; ++t) {
            noc_async_read_tile(b * blk + t, x, l1 + t * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb, blk);
        if (b % 4 == 0) {
            SE_MARK("SB_XREAD");
        }
    }
}
