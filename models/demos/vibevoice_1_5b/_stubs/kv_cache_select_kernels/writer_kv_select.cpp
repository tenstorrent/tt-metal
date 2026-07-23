// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the fused KV-cache select kernel: writes the selected tile straight
// back into the SAME cache DRAM buffer it was read from (in-place update), in the
// identical (kv, row_tile, col_tile) order the reader used.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t KV = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);

    constexpr auto cb_out = tt::CBIndex::c_7;
    constexpr auto cache_args = TensorAccessorArgs<3>();
    const auto cache_acc = TensorAccessor(cache_args, cache_addr);

    Noc noc;
    CircularBuffer cb_out_buf(cb_out);
    const uint32_t tile_bytes = cb_out_buf.get_tile_size();

    for (uint32_t kv = 0; kv < KV; ++kv) {
        for (uint32_t rt = 0; rt < RT; ++rt) {
            for (uint32_t ct = 0; ct < CT; ++ct) {
                const uint32_t cache_tile_id = kv * RT * CT + rt * CT + ct;

                cb_out_buf.wait_front(1);
                noc.async_write(cb_out_buf, cache_acc, tile_bytes, {}, {.page_id = cache_tile_id});
                noc.async_write_barrier();
                cb_out_buf.pop_front(1);
            }
        }
    }
}
