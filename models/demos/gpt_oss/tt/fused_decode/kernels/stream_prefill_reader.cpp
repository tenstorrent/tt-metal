// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill matmul on a decode stream weight, reader (NCRISC, NOC0) (tt/stream_prefill_linear.py StreamPrefillLinear).
// This core owns `cols` consecutive slots of one DRAM bank of the stream (each a column of KT = KX + 1 tiles: KX
// weight tiles, then the bias tile; one contiguous byte range) and the output row tiles [m0, m0 + m_cnt). Per M block
// of MB row tiles: the x rows into cb_x ([row][KT] tiles: the row's KX x tiles from the tensor accessor, then the
// bias-multiplier "ones" tile), then the weight columns into cb_w in groups of G columns ([column][KT] tiles, one read
// per group). RESIDENT: one group holds all of this core's columns; it is read for the first M block only and stays in
// cb_w (stream_prefill_compute.cpp pops it at the end). Otherwise cb_w is a two-group ring, refilled every M block.
// Every push is a whole block (MB rows / G columns) so the ring never holds a straddling block.
//
// common runtime args: [w_addr, x_addr, ones_addr]
// runtime args: [bank_id, w_offset (bytes of the core's first slot in the bank), cols, m0, m_cnt]
// compile-time args: [cb_x, cb_w, KX, w_tile_bytes, MB, G, RESIDENT, accessors (x, ones)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ones_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t w_offset = get_arg_val<uint32_t>(1);
    const uint32_t cols = get_arg_val<uint32_t>(2);
    const uint32_t m0 = get_arg_val<uint32_t>(3);
    const uint32_t m_cnt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t KX = get_compile_time_arg_val(2);
    constexpr uint32_t w_tile = get_compile_time_arg_val(3);
    constexpr uint32_t MB = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr bool RESIDENT = get_compile_time_arg_val(6) != 0;
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto ones_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t KT = KX + 1;
    constexpr uint32_t col_bytes = KT * w_tile;

    const uint32_t x_tile = get_tile_size(cb_x);
    const auto s_x = TensorAccessor(x_args, x_addr, x_tile);
    const auto s_ones = TensorAccessor(ones_args, ones_addr, x_tile);
    const uint64_t ones_noc = s_ones.get_noc_addr(0);
    const uint32_t n_groups = (cols + G - 1) / G;

    bool loaded = false;
    for (uint32_t mb = 0; mb < m_cnt; mb += MB) {
        const uint32_t rows = m_cnt - mb < MB ? m_cnt - mb : MB;
        cb_reserve_back(cb_x, MB * KT);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t page0 = (m0 + mb + r) * KX;
            for (uint32_t k = 0; k < KX; ++k) {
                noc_async_read(s_x.get_noc_addr(page0 + k), l1, x_tile);
                l1 += x_tile;
            }
            noc_async_read(ones_noc, l1, x_tile);
            l1 += x_tile;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, MB * KT);
        if (RESIDENT && loaded) {
            continue;
        }
        for (uint32_t g = 0; g < n_groups; ++g) {
            const uint32_t gc = cols - g * G < G ? cols - g * G : G;
            cb_reserve_back(cb_w, G * KT);
            const uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, w_addr + w_offset + g * G * col_bytes);
            noc_async_read(src, get_write_ptr(cb_w), gc * col_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_w, G * KT);
        }
        loaded = true;
    }
}
