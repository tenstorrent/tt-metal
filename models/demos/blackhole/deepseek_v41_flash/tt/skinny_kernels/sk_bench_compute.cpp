// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// compute-only throughput benchmark of custom_mm (tiny in0 tiles, bfp8 weights, LoFi): R repetitions of NG column
// groups of CT columns x KT K tiles, MT row tiles. The data are whatever is in L1 (no producer): only the cycle counts
// are meaningful.  CT args: 0 cb_a, 1 cb_w, 2 cb_o, 3 KT, 4 CT, 5 NG, 6 MT, 7 R, 8 fp32, 9 ring tiles, 10 mode (0 =
// ct_dim loop per column, 1 = ct_dim block) Zone B_LOOP = the whole loop.

#include <cstdint>
#include "sk_mm.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    constexpr uint32_t NG = get_compile_time_arg_val(5);
    constexpr uint32_t MT = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);
    constexpr bool fp32 = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t RING = get_compile_time_arg_val(9);
    constexpr uint32_t mode = get_compile_time_arg_val(10);

    deepseek_compute_kernel_init();
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    if constexpr (fp32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, fp32>(cb_a, cb_w, cb_o, mode == 1 ? CT : 1);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(cb_a, cb_w);
        pack_reconfig_data_format<true>(cb_o);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_a, cb_w, cb_o, mode == 1 ? CT : 1);
    }
    pack_block_contiguous_init(cb_o);
    {
        DeviceZoneScopedN("B_LOOP");
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t g = 0; g < NG; ++g) {
                cb_reserve_back(cb_o, MT * CT);
                tile_regs_acquire();
                const uint32_t w0 = (g * KT * CT) % (RING - KT * CT + 1);
                if constexpr (mode == 1) {
                    for (uint32_t m = 0; m < MT; ++m) {
                        custom_mm_block<true>(cb_a, cb_w, m * KT, w0, m * CT, KT, CT);
                    }
                } else {
                    for (uint32_t c = 0; c < CT; ++c) {
                        for (uint32_t m = 0; m < MT; ++m) {
                            custom_mm_block<true>(cb_a, cb_w, m * KT, w0 + c * KT, m * CT + c, KT);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_block_contiguous(0, cb_o, MT * CT);
                tile_regs_release();
                cb_push_back(cb_o, MT * CT);
                cb_wait_front(cb_o, MT * CT);
                cb_pop_front(cb_o, MT * CT);
            }
        }
    }
    custom_mm_block_uninit<dense_packing>();
}
