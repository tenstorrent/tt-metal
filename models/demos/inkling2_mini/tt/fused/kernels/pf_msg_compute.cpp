// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill cross-pass messages, compute. Per tile (R, j) and message k < 4:
//   m_k = n1 + diff * w_k   as ttnn.add(n1, ttnn.multiply(diff, w[:, c0 + k])) on fp32 tensors: the binary_ng
//   SFPU ops (mul_binary_tile, then add_binary_tile with RNE narrowing; fp32 DST, so the product stays fp32).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_n = 0, cb_d = 1, cb_w = 2, cb_out = 16;
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_d, cb_out);
    copy_init(cb_d);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t nj = get_arg_val<uint32_t>(1 + u);
        cb_wait_front(cb_w, 4);
        for (uint32_t j = 0; j < nj; ++j) {
            cb_wait_front(cb_n, 1);
            cb_wait_front(cb_d, 1);
            for (uint32_t k = 0; k < 4; ++k) {
                tile_regs_acquire();
                copy_tile(cb_d, 0, 0);
                copy_tile(cb_w, k, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                copy_tile(cb_n, 0, 2);
                add_binary_tile_init();
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(2, 0, 2);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_out, 1);
                pack_tile(2, cb_out);
                cb_push_back(cb_out, 1);
                tile_regs_release();
            }
            cb_pop_front(cb_n, 1);
            cb_pop_front(cb_d, 1);
        }
        cb_pop_front(cb_w, 4);
    }
}
