// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 EP combine — compute (moe_combine_common.hpp): y = sum_j slot_j * bcast_col(w_j), exact fp32 SFPU products /
// sums (fp32 DEST, fp32 unpack-to-dest of the weight tiles), packed bf16. Compile-time: K, CTC. Runtime: first unit,
// unit count.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

#include "moe_combine_common.hpp"

using namespace mcomb;

namespace {

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);  // bf16 slot tiles / fp32 weight tiles alternate
    copy_init(cb);
    copy_tile(cb, i, d);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t CTC = get_compile_time_arg_val(1);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_S, CB_Y);
    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + n; ++u) {
        const uint32_t r = u / CTC;
        if (r != cur_r) {
            if (cur_r != 0xFFFFFFFFu) {
                cb_pop_front(CB_W, K);
            }
            cur_r = r;
            cb_wait_front(CB_W, K);
        }
        cb_wait_front(CB_S, K);
        cb_reserve_back(CB_Y, 1);
        tile_regs_acquire();
        load(CB_S, 0, 0);
        load(CB_W, 0, 2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        for (uint32_t j = 1; j < K; ++j) {
            load(CB_S, j, 1);
            load(CB_W, j, 2);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(1, 2);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_Y);
        pack_tile(0, CB_Y);
        tile_regs_release();
        cb_push_back(CB_Y, 1);
        cb_pop_front(CB_S, K);
    }
    cb_pop_front(CB_W, K);
}
