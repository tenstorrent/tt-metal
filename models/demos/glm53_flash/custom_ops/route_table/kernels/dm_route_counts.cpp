// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Route table, pass 1 (tt-blaze G53F-0195): per 32-token block, how many of the block's picks go to each of the chip's
// local experts. A pick (t, j) is local when its routing weight is a nonzero normal fp32 (the router's dense matrix is
// bf16(w) at the picks, nonzero exactly there; subnormals flush) and its expert e = idx[t, j] lies in the chip's
// contiguous expert range (the dispatch table is >= 0 there). Reads idx / w straight from their fp32 TILE layout
// (one tile per block: element (r, j < 16) at face r < 16 ? 0 : 2, offset (r % 16) * 16 + j) and writes one
// [E_LOC] uint32 row per block (padded to the row's aligned page: 64 B multiples on Blackhole DRAM).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t K = get_compile_time_arg_val(0);
constexpr uint32_t E = get_compile_time_arg_val(1);
constexpr uint32_t E_LOC = get_compile_time_arg_val(2);
constexpr uint32_t cb_scr_id = get_compile_time_arg_val(3);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
constexpr uint32_t table_bytes = get_compile_time_arg_val(5);
constexpr uint32_t row_bytes = get_compile_time_arg_val(6);
constexpr auto idx_args = TensorAccessorArgs<7>();
constexpr auto w_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
constexpr auto table_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto out_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr uint32_t TILE_ROWS = 32;

// the fp32 bit pattern of a non-negative integer value below 2^24 -> that integer (exact; no FPU on the data movers)
FORCE_INLINE uint32_t f32_to_u32(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e < 127u) {
        return 0u;
    }
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    return e >= 150u ? (m << (e - 150u)) : (m >> (150u - e));
}

FORCE_INLINE uint32_t tile_off(uint32_t r, uint32_t j) { return (r < 16u ? 0u : 512u) + (r & 15u) * 16u + j; }

void kernel_main() {
    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t block0 = get_arg_val<uint32_t>(4);
    const uint32_t nblocks = get_arg_val<uint32_t>(5);

    const auto s_idx = TensorAccessor(idx_args, idx_addr, tile_bytes);
    const auto s_w = TensorAccessor(w_args, w_addr, tile_bytes);
    const auto s_table = TensorAccessor(table_args, table_addr, table_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, row_bytes);

    Noc noc;
    CircularBuffer cb_scr(cb_scr_id);
    cb_scr.reserve_back(1);  // private scratch, reserved once and never pushed
    const uint32_t base = get_write_ptr(cb_scr_id);
    const uint32_t idx_l1 = base, w_l1 = base + tile_bytes, table_l1 = base + 2 * tile_bytes;
    const uint32_t row_l1 = table_l1 + table_bytes;
    const CoreLocalMem<volatile uint32_t> idx_t(idx_l1), w_t(w_l1), row(row_l1);
    const CoreLocalMem<volatile int32_t> table(table_l1);

    noc.async_read(s_table, CoreLocalMem<uint32_t>(table_l1), table_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    uint32_t lo = 0;
    while (lo < E && table[lo] < 0) {
        ++lo;
    }

    for (uint32_t b = block0; b < block0 + nblocks; ++b) {
        noc.async_read(s_idx, CoreLocalMem<uint32_t>(idx_l1), tile_bytes, {.page_id = b}, {});
        noc.async_read(s_w, CoreLocalMem<uint32_t>(w_l1), tile_bytes, {.page_id = b}, {});
        noc.async_write_barrier();  // the previous block's row has left L1
        for (uint32_t l = 0; l < row_bytes / 4; ++l) {
            row[l] = 0u;
        }
        noc.async_read_barrier();
        for (uint32_t r = 0; r < TILE_ROWS; ++r) {
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t o = tile_off(r, j);
                if (((w_t[o] >> 23) & 0xFFu) == 0u) {
                    continue;  // zero (or subnormal) weight: not a pick of this chip
                }
                const uint32_t l = f32_to_u32(idx_t[o]) - lo;  // unsigned: below the range wraps high
                if (l < E_LOC) {
                    row[l] = row[l] + 1u;
                }
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(row_l1), s_out, row_bytes, {}, {.page_id = b});
    }
    noc.async_write_barrier();
}
