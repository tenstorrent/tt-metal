// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill skewed rel bias, compute. Per unit and block: tilize the 32 x 32 rel-state block (rows = query tiles),
// multiply it by the 33 tiles of B_r (one K tile, HiFi4, fp32 dest, like the ttnn.matmul it replaces) and pack the
// 32 x 1056 result untilized (row m = the skewed rel row of query 32 (32 i + m) + r).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("t");
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");
    constexpr uint32_t BW = get_named_compile_time_arg_val("bw");  // output tiles per dest block (NB % BW == 0)
    constexpr uint32_t QT = T / 32, NA = (QT + 31) / 32;
    constexpr uint32_t cb_rm = 0, cb_b = 1, cb_a = 2, cb_out = 16;
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_rm, cb_b, cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        cb_wait_front(cb_b, NB);
        for (uint32_t i = 0; i < NA; ++i) {
            cb_wait_front(cb_rm, 1);
            cb_reserve_back(cb_a, 1);
            reconfig_data_format_srca(cb_rm);
            pack_reconfig_data_format(cb_a);
            tilize_init(cb_rm, 1, cb_a);
            tilize_block(cb_rm, 1, cb_a);
            tilize_uninit(cb_rm, cb_a);
            cb_push_back(cb_a, 1);
            cb_pop_front(cb_rm, 1);

            cb_wait_front(cb_a, 1);
            cb_reserve_back(cb_out, NB);
            reconfig_data_format(cb_b, cb_a);  // matmul: in1 -> SrcA, in0 -> SrcB
            pack_reconfig_data_format(cb_out);
            matmul_init(cb_a, cb_b);
            pack_untilize_dest_init<BW, NB>(cb_out);
            for (uint32_t blk = 0; blk < NB / BW; ++blk) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < BW; ++w) {
                    matmul_tiles(cb_a, cb_b, 0, blk * BW + w, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<BW, NB>(cb_out, 1, blk);
                tile_regs_release();
            }
            pack_untilize_uninit(cb_out);
            cb_push_back(cb_out, NB);
            cb_pop_front(cb_a, 1);
        }
        cb_pop_front(cb_b, NB);
    }
}
