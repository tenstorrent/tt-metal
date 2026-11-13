// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill matmul on a decode stream weight, compute (tt/stream_prefill_linear.py StreamPrefillLinear). Per M block
// (cb_x: [row][KT] tiles, the last tile of a row the bias multiplier; stream_prefill_reader.cpp) and per weight group
// (cb_w: [column][KT] tiles, the last tile of a column the bias tile): for each column, blocks of up to RT rows x 1
// column summed over all KT K tiles in DST (FP32), packed one tile per push (row order within the block).
//
// runtime args: [cols, m_cnt]
// compile-time args: [cb_x, cb_w, cb_out, KT, MB, G, RESIDENT]

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t MB = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr bool RESIDENT = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t RT = 4;  // rows per DST block (FP32: 4 tiles)
    const uint32_t cols = get_arg_val<uint32_t>(0);
    const uint32_t m_cnt = get_arg_val<uint32_t>(1);
    const uint32_t n_groups = (cols + G - 1) / G;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    // The matmul is set up for a block of rt rows; only packs run between blocks, so it is set up again only when rt
    // changes (the last, partial block of an M block).
    uint32_t init_rt = 0;
    bool loaded = false;
    for (uint32_t mb = 0; mb < m_cnt; mb += MB) {
        const uint32_t rows = m_cnt - mb < MB ? m_cnt - mb : MB;
        cb_wait_front(cb_x, MB * KT);
        for (uint32_t g = 0; g < n_groups; ++g) {
            const uint32_t gc = cols - g * G < G ? cols - g * G : G;
            if (!RESIDENT || !loaded) {
                cb_wait_front(cb_w, G * KT);
            }
            for (uint32_t c = 0; c < gc; ++c) {
                for (uint32_t r0 = 0; r0 < rows; r0 += RT) {
                    const uint32_t rt = rows - r0 < RT ? rows - r0 : RT;
                    if (rt != init_rt) {
                        matmul_block_init(cb_x, cb_w, 0, 1, rt, KT);
                        init_rt = rt;
                    }
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < KT; ++k) {
                        matmul_block(cb_x, cb_w, r0 * KT + k, c * KT + k, 0, false, 1, rt, KT);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    // one tile per push (several per push corrupted one landing on the CB's last page, gmoe_*.cpp)
                    for (uint32_t r = 0; r < rt; ++r) {
                        cb_reserve_back(cb_out, 1);
                        pack_tile(r, cb_out);
                        cb_push_back(cb_out, 1);
                    }
                    tile_regs_release();
                }
            }
            if (!RESIDENT) {
                cb_pop_front(cb_w, G * KT);
            }
        }
        loaded = true;
        cb_pop_front(cb_x, MB * KT);
    }
    if (RESIDENT && loaded) {
        cb_pop_front(cb_w, G * KT);
    }
}
