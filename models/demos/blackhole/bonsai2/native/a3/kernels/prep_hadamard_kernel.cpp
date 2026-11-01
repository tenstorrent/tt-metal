// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A3 input prep: x' = H(D x) per 1024-block, scattered straight into the A3 workers'
// block-masked input buffer.
//
// The A3 input is a persistent [8, K] buffer of 8x32 tiles on every A3 worker, where
// K-tile k carries x'[k] in row (k/4) % 8 and zeros elsewhere. Row r of tile k is always
// the same row, so the zero rows are written once at allocation and never again; this
// kernel only writes the one live row per tile.
//
// Work unit (b, r): block b, output K-tiles k = 32 b + j for j in [4r, 4r+4), all of
// which land in row r. M_b = diag(d_b) H / 32 is stored bfp2 (entries +-1/32, exact) as
// [K, 1024] tiles; x'[32b + j] = x_b @ M_b[:, j].
//
// NCRISC: x_b row -> 32 1x32 tiles (cb 0), M_b column sticks (cb 1).
// TRISC:  custom_mm over 32 tiles per output, LoFi (exact for +-1/32), fp32 DST.
// BRISC:  each output row (64 B) -> 2 x 32 B writes into every worker's buffer.
//
// Runtime args, per core:
//   NCRISC: x_addr, m_addr, num_units, then (b, r) per unit
//   BRISC:  x8_addr, num_units, (b, r) per unit, num_workers, then noc (x << 16 | y) per worker
// Compile time: src_tile32 (1: x is [.., 32, K] bf16 32x32 tiles, row 0 live;
//   0: x is [1, K] 1x32 tiles), x_is_dram, m_page (bytes of one bfp2 tile).

#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/pack_block.h"
#endif

constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_m = 1;
constexpr uint32_t cb_o = 2;
constexpr uint32_t KT = 32;  // K-tiles per 1024-block

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr bool src_tile32 = get_named_compile_time_arg_val("pp_src_tile32") == 1;
    constexpr bool x_is_dram = get_named_compile_time_arg_val("pp_x_is_dram") == 1;
    constexpr uint32_t m_page = get_named_compile_time_arg_val("pp_m_page");
    constexpr uint32_t dbg = get_named_compile_time_arg_val("pp_debug");  // 1: skip M reads, 2: skip x reads
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<x_is_dram> xg{.bank_base_address = x_addr, .page_size = src_tile32 ? 2048u : 64u};
    constexpr uint32_t upb = get_named_compile_time_arg_val("pp_units_per_bank");
    constexpr uint32_t unit_bytes = 4 * KT * m_page;
    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t b = get_arg_val<uint32_t>(3 + 2 * u);
        const uint32_t r = get_arg_val<uint32_t>(4 + 2 * u);
        cb_reserve_back(cb_x, KT);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < KT && !(dbg & 2); ++i) {
            const uint32_t page = b * KT + i;
            if constexpr (src_tile32) {
                // row 0 of a 32x32 bf16 tile: face 0 row 0 at +0, face 1 row 0 at +512
                noc_async_read(get_noc_addr(page, xg, 0), l1, 32);
                noc_async_read(get_noc_addr(page, xg, 512), l1 + 32, 32);
            } else {
                noc_async_read(get_noc_addr(page, xg, 0), l1, 64);
            }
            l1 += 64;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, KT);
        // M for this unit: one contiguous run in bank u / upb (see hadamard_tensor)
        const uint32_t unit = 8 * b + r;
        uint32_t src = m_addr + (unit % upb) * unit_bytes;
        const uint32_t bank = unit / upb;
        for (uint32_t jj = 0; jj < 4; ++jj) {
            cb_reserve_back(cb_m, KT);
            if constexpr (!(dbg & 1)) {
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), get_write_ptr(cb_m), KT * m_page);
                noc_async_read_barrier();
            }
            src += KT * m_page;
            cb_push_back(cb_m, KT);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t x8_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t nw_idx = 2 + 2 * num_units;
    const uint32_t num_workers = get_arg_val<uint32_t>(nw_idx);
    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t b = get_arg_val<uint32_t>(2 + 2 * u);
        const uint32_t r = get_arg_val<uint32_t>(3 + 2 * u);
        for (uint32_t j = 4 * r; j < 4 * r + 4; ++j) {
            const uint32_t k = b * KT + j;
            const uint32_t dst0 = x8_addr + k * 512 + r * 32;  // 8x32 bf16 tile: face 0 row r
            cb_wait_front(cb_o, 1);
            const uint32_t src = get_read_ptr(cb_o);
            for (uint32_t w = 0; w < num_workers; ++w) {
                const uint32_t xy = get_arg_val<uint32_t>(nw_idx + 1 + w);
                noc_async_write(src, get_noc_addr(xy >> 16, xy & 0xffff, dst0), 32);
                noc_async_write(src + 32, get_noc_addr(xy >> 16, xy & 0xffff, dst0 + 256), 32);  // face 1 row r
            }
            noc_async_write_barrier();
            cb_pop_front(cb_o, 1);
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    custom_mm_block_init<false, CustomMmDestLayout::Default, true>(cb_x, cb_m, cb_o);
    pack_block_contiguous_init(cb_o);
    for (uint32_t u = 0; u < num_units; ++u) {
        cb_wait_front(cb_x, KT);
        for (uint32_t j = 0; j < 4; ++j) {
            cb_wait_front(cb_m, KT);
            tile_regs_acquire();
            custom_mm_block<CustomMmDestLayout::Default, false>(cb_x, cb_m, 0, 0, 0, KT);
            tile_regs_commit();
            cb_pop_front(cb_m, KT);
            tile_regs_wait();
            cb_reserve_back(cb_o, 1);
            pack_block_contiguous(0, cb_o, 1);
            cb_push_back(cb_o, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_x, KT);
    }
#endif
}
