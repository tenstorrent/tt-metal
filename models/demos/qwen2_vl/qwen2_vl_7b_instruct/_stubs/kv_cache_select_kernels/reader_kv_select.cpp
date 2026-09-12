// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the fused KV-cache select kernel:
//   cache_new = cache + onehot * (new_val - cache)
//
// Streams, in a fixed (kv, row_tile, col_tile) order, one tile each of:
//   - cache   [1, KV, C, HD]  (KV*RT*CT distinct tiles)
//   - onehot  [1, 1,  C, 1 ]  (RT distinct tiles, reused across kv/col_tile)
//   - new_val [1, KV, 1, HD]  (KV*CT distinct tiles, reused across row_tile)
// into three circular buffers, for the compute kernel to consume 1:1:1.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t onehot_addr = get_arg_val<uint32_t>(1);
    const uint32_t new_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t KV = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);

    constexpr auto cb_cache = tt::CBIndex::c_0;
    constexpr auto cb_onehot = tt::CBIndex::c_1;
    constexpr auto cb_new = tt::CBIndex::c_2;

    constexpr auto cache_args = TensorAccessorArgs<3>();
    constexpr auto onehot_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto new_args = TensorAccessorArgs<onehot_args.next_compile_time_args_offset()>();

    const auto cache_acc = TensorAccessor(cache_args, cache_addr);
    const auto onehot_acc = TensorAccessor(onehot_args, onehot_addr);
    const auto new_acc = TensorAccessor(new_args, new_addr);

    Noc noc;
    CircularBuffer cb_cache_buf(cb_cache);
    CircularBuffer cb_onehot_buf(cb_onehot);
    CircularBuffer cb_new_buf(cb_new);

    const uint32_t cache_tile_bytes = cb_cache_buf.get_tile_size();
    const uint32_t onehot_tile_bytes = cb_onehot_buf.get_tile_size();
    const uint32_t new_tile_bytes = cb_new_buf.get_tile_size();

    for (uint32_t kv = 0; kv < KV; ++kv) {
        for (uint32_t rt = 0; rt < RT; ++rt) {
            for (uint32_t ct = 0; ct < CT; ++ct) {
                const uint32_t cache_tile_id = kv * RT * CT + rt * CT + ct;
                const uint32_t onehot_tile_id = rt;
                const uint32_t new_tile_id = kv * CT + ct;

                cb_cache_buf.reserve_back(1);
                cb_onehot_buf.reserve_back(1);
                cb_new_buf.reserve_back(1);

                noc.async_read(
                    cache_acc, cb_cache_buf, cache_tile_bytes, {.page_id = cache_tile_id}, {.offset_bytes = 0});
                noc.async_read(
                    onehot_acc, cb_onehot_buf, onehot_tile_bytes, {.page_id = onehot_tile_id}, {.offset_bytes = 0});
                noc.async_read(new_acc, cb_new_buf, new_tile_bytes, {.page_id = new_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();

                cb_cache_buf.push_back(1);
                cb_onehot_buf.push_back(1);
                cb_new_buf.push_back(1);
            }
        }
    }
}
