// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE weight relayout (init, dflash/grouped_moe.py GroupedMoE._relayout): every work unit's weight tiles into
// one contiguous DRAM page, so the grouped readers fetch a unit in one large read (tile by tile from the dense layout:
// ~180 GB/s for the gate|up pass).
//   gate|up (default): unit u = (e, j), e = u / NU, j = u % NU, column n = g0(e) + j: [k][gate (k, n), up (k, n)];
//   down (GMOE_RELAYOUT_DOWN): unit u = (e, c), c = u % NU: [k] down (g0(e) + k, c), the shared K tile (e even: its
//   last, e odd: its first) from `edge` (row tile e: the neighbour's rows zeroed).
// GMOE_RELAYOUT_SPLIT: each unit's page stored as that many consecutive smaller pages (spread over the DRAM banks).
// Core i of NC takes units i, i + NC, ...
//
// common runtime args: [src0, src1, dst]; compile-time args: [cb_buf, NC, GX, UNITS, KT, NT, NU, KD, accessors
// (src0, src1, dst)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"

void kernel_main() {
    const uint32_t src0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_buf = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t GX = get_compile_time_arg_val(2);
    constexpr uint32_t UNITS = get_compile_time_arg_val(3);
    constexpr uint32_t KT = get_compile_time_arg_val(4);  // gate|up K tiles
    constexpr uint32_t NT = get_compile_time_arg_val(5);  // tile columns of src0 (and src1)
    constexpr uint32_t NU = get_compile_time_arg_val(6);  // units per expert
    constexpr uint32_t KD = get_compile_time_arg_val(7);  // down K tiles per expert
    constexpr auto s0_args = TensorAccessorArgs<8>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    constexpr uint32_t w_tile = 576;  // BFP4_b
#ifdef GMOE_RELAYOUT_DOWN
    constexpr uint32_t page = KD * w_tile;
#else
    constexpr uint32_t page = 2 * KT * w_tile;
#endif
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const auto s0 = TensorAccessor(s0_args, src0_addr, w_tile);
    const auto s1 = TensorAccessor(s1_args, src1_addr, w_tile);
#ifdef GMOE_RELAYOUT_SPLIT
    const auto d = TensorAccessor(d_args, dst_addr, page / GMOE_RELAYOUT_SPLIT);
#else
    const auto d = TensorAccessor(d_args, dst_addr, page);
#endif

    cb_reserve_back(cb_buf, 1);
    const uint32_t buf = get_write_ptr(cb_buf);
    for (uint32_t u = c; u < UNITS; u += NC) {
        const uint32_t e = u / NU, j = u % NU;
#ifdef GMOE_RELAYOUT_DOWN
        const uint32_t edge_k = (e % 2) ? 0 : KD - 1;
        for (uint32_t k = 0; k < KD; ++k) {
            const uint64_t src =
                k == edge_k ? s1.get_noc_addr(e * NT + j) : s0.get_noc_addr((gmoe::g0(e) + k) * NT + j);
            noc_async_read(src, buf + k * w_tile, w_tile);
        }
#else
        const uint32_t n = gmoe::g0(e) + j;
        for (uint32_t k = 0; k < KT; ++k) {
            noc_async_read(s0.get_noc_addr(k * NT + n), buf + (2 * k) * w_tile, w_tile);
            noc_async_read(s1.get_noc_addr(k * NT + n), buf + (2 * k + 1) * w_tile, w_tile);
        }
#endif
        noc_async_read_barrier();
#ifdef GMOE_RELAYOUT_SPLIT
        // the unit's page as GMOE_RELAYOUT_SPLIT consecutive pages (u SPLIT + q: spread over the DRAM banks)
        constexpr uint32_t sp = page / GMOE_RELAYOUT_SPLIT;
        for (uint32_t q = 0; q < GMOE_RELAYOUT_SPLIT; ++q) {
            noc_async_write(buf + q * sp, d.get_noc_addr(u * GMOE_RELAYOUT_SPLIT + q), sp);
        }
#else
        noc_async_write(buf, d.get_noc_addr(u), page);
#endif
        noc_async_write_barrier();
    }
}
