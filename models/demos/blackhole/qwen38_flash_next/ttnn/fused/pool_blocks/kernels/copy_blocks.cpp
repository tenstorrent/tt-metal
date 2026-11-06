// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// pool_blocks copy_blocks (data movement only): whole vLLM blocks of the paged K/V pool [1, 1, NB * bs, 512] (ROW_MAJOR
// bf16, 1024-byte rows, block b = rows [b * bs, (b + 1) * bs)) and of the index pool [NB, 1, bs / 4, 128] (TILE bf16,
// block b = the tile ids [b * bs / 32, (b + 1) * bs / 32)).  Pair (src, dst) copies block src to block dst in both
// pools; src == 0xFFFFFFFF zero-fills block dst.  A unit is 32 K/V rows plus one index tile; pair p's unit u (u < bs /
// 32) is the global unit p * (bs / 32) + u, and core j of n takes the global units j, j + n, ...  The host guarantees
// that the dst blocks are distinct and none is the src of another pair, so units never race. Compile-time args:
// TensorAccessorArgs kv pool, index pool. Runtime args: 0 kv pool, 1 index pool addresses, 2 pair count, 3 log2(bs), 4
// core index, 5 core count, then (src, dst) per pair.  CB 0: one 64-byte aligned region = the unit buffer (32 rows +
// one tile), then a zero unit.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t kv_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_addr = get_arg_val<uint32_t>(1);
    const uint32_t pairs = get_arg_val<uint32_t>(2);
    const uint32_t bs_shift = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    const uint32_t cores = get_arg_val<uint32_t>(5);
    constexpr uint32_t ROW_BYTES = 1024, GROUP_ROWS = 32, TILE_BYTES = 2048,
                       UNIT_BYTES = GROUP_ROWS * ROW_BYTES + TILE_BYTES;
    constexpr uint32_t ZERO_SOURCE = 0xFFFFFFFF, ZERO_CHUNK = 8192;
    constexpr auto kv_args = TensorAccessorArgs<0>();
    constexpr auto index_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto index = TensorAccessor(index_args, index_addr);

    const uint32_t units_per_pair = 1u << (bs_shift - 5);
    const uint32_t units = pairs * units_per_pair;
    if (core >= units) {
        return;
    }
    cb_reserve_back(0, 1);
    const uint32_t buffer = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t zeros = buffer + UNIT_BYTES;
    const uint32_t tiles_shift = bs_shift - 5;

    bool any_zero = false;
    for (uint32_t p = 0; p < pairs; ++p) {
        any_zero |= get_arg_val<uint32_t>(6 + 2 * p) == ZERO_SOURCE;
    }
    if (any_zero) {
        Noc noc;
        for (uint32_t offset = 0; offset < UNIT_BYTES; offset += ZERO_CHUNK) {
            const uint32_t bytes = UNIT_BYTES - offset < ZERO_CHUNK ? UNIT_BYTES - offset : ZERO_CHUNK;
            noc.async_write_zeros(CoreLocalMem<uint32_t>(zeros + offset), bytes);
            noc.write_zeros_l1_barrier();
        }
    }

    for (uint32_t g = core; g < units; g += cores) {
        const uint32_t p = g >> tiles_shift;
        const uint32_t u = g & (units_per_pair - 1);
        const uint32_t src = get_arg_val<uint32_t>(6 + 2 * p);
        const uint32_t dst = get_arg_val<uint32_t>(7 + 2 * p);
        const uint32_t dst_row = (dst << bs_shift) + u * GROUP_ROWS;
        const uint32_t dst_tile = (dst << tiles_shift) + u;
        uint32_t out = buffer;
        if (src == ZERO_SOURCE) {
            out = zeros;
        } else {
            const uint32_t src_row = (src << bs_shift) + u * GROUP_ROWS;
            const uint32_t src_tile = (src << tiles_shift) + u;
            for (uint32_t k = 0; k < GROUP_ROWS; ++k) {
                noc_async_read(kv.get_noc_addr(src_row + k, 0), buffer + k * ROW_BYTES, ROW_BYTES);
            }
            noc_async_read_page(src_tile, index, buffer + GROUP_ROWS * ROW_BYTES);
            noc_async_read_barrier();
        }
        for (uint32_t k = 0; k < GROUP_ROWS; ++k) {
            noc_async_write(out + k * ROW_BYTES, kv.get_noc_addr(dst_row + k, 0), ROW_BYTES);
        }
        noc_async_write_page(dst_tile, index, out + GROUP_ROWS * ROW_BYTES);
        noc_async_write_barrier();
    }
}
