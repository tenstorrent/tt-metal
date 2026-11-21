// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill matmul on a decode stream weight, compute (tt/stream_prefill_linear.py StreamPrefillLinear). Per M block
// (cb_x: [row][KT] tiles, the last tile of a row the bias multiplier; stream_prefill_reader.cpp) and per weight group
// (cb_w: [KT][G] tiles, tile k of column c at k * G + c; the last K tile of a column the bias tile): DST blocks of
// rt rows x ct consecutive columns (rt * ct <= 4, FP32 DST) summed over all KT K tiles, column blocks first, then row
// blocks; each block packed one tile per push in row-major order. rt = min(rows of the M block, 4), ct = 4 / rt
// (stream_prefill_linear.py block_shape / dst_blocks): an M block of 1 or 2 rows (K = 2880 fills L1) still fills DST.
//
// compile-time args: [cb_x, cb_w, cb_out, KT, MB, G, RESIDENT, cols, m_cnt] (one kernel per (cols, m_cnt) of the
// plan's cores: no runtime args, stream_prefill_args.hpp)

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
    constexpr uint32_t DST_TILES = 4;  // FP32 DST, half sync
    constexpr uint32_t cols = get_compile_time_arg_val(7);
    constexpr uint32_t m_cnt = get_compile_time_arg_val(8);
    const uint32_t n_groups = (cols + G - 1) / G;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    // The matmul is set up for a block shape; only packs run between blocks, so it is set up again only when the
    // shape changes (the partial blocks at the ends of an M block / weight group).
    uint32_t init_rt = 0;
    uint32_t init_ct = 0;
    bool loaded = false;
    for (uint32_t mb = 0; mb < m_cnt; mb += MB) {
        const uint32_t rows = m_cnt - mb < MB ? m_cnt - mb : MB;
        const uint32_t rt_m = rows < DST_TILES ? rows : DST_TILES;
        const uint32_t ct_m = DST_TILES / rt_m;
        cb_wait_front(cb_x, MB * KT);
        for (uint32_t g = 0; g < n_groups; ++g) {
            const uint32_t gc = cols - g * G < G ? cols - g * G : G;
            if (!RESIDENT || !loaded) {
                cb_wait_front(cb_w, G * KT);
            }
            for (uint32_t c0 = 0; c0 < gc; c0 += ct_m) {
                const uint32_t ct = gc - c0 < ct_m ? gc - c0 : ct_m;
                for (uint32_t r0 = 0; r0 < rows; r0 += rt_m) {
                    const uint32_t rt = rows - r0 < rt_m ? rows - r0 : rt_m;
                    if (rt != init_rt || ct != init_ct) {
                        matmul_block_init(cb_x, cb_w, 0, ct, rt, KT);
                        init_rt = rt;
                        init_ct = ct;
                    }
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < KT; ++k) {
                        matmul_block(cb_x, cb_w, r0 * KT + k, k * G + c0, 0, false, ct, rt, KT);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    // one tile per push (several per push corrupted one landing on the CB's last page, gmoe_*.cpp)
                    for (uint32_t i = 0; i < rt * ct; ++i) {
                        cb_reserve_back(cb_out, 1);
                        pack_tile(i, cb_out);
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
