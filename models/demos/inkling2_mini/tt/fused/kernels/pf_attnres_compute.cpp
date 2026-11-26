// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill multi-row attn-res, compute (fp32 SFPU, the op chain of prefill_model._AttnRes in the same order):
//   m = max_d lg_d; e_d = exp(lg_d - m); z = e_0 + e_1 + ...; rz = 1 / z; w_d = e_d * rz      (per unit, small tiles)
//   out_q[:, j] = sum_d n_d[:, j] * w_d[:, q]   (w_d[:, q] broadcast across the row: exact fp32 column broadcast)
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t D1 = get_named_compile_time_arg_val("d1");
    constexpr uint32_t NQ = get_named_compile_time_arg_val("nq");
    constexpr uint32_t cb_lg = 0, cb_e = 1, cb_w = 2, cb_wc = 3, cb_b = 4, cb_n = 5, cb_m = 6, cb_rz = 7, cb_out = 16;
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_lg, cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t ntiles = get_arg_val<uint32_t>(1 + u);  // column tiles of this unit
        cb_wait_front(cb_lg, D1);
        // m = max over depth
        tile_regs_acquire();
        load(cb_lg, 0, 0);
        binary_max_tile_init();
        for (uint32_t d = 1; d < D1; ++d) {
            load(cb_lg, d, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_m);
        tile_regs_release();
        // e_d = exp(lg_d - m)
        cb_wait_front(cb_m, 1);
        for (uint32_t d = 0; d < D1; ++d) {
            tile_regs_acquire();
            load(cb_lg, d, 1);
            load(cb_m, 0, 0);
            sub_binary_tile_init();
            sub_binary_tile(1, 0, 1);
            exp_tile_init();
            exp_tile(1);
            tile_regs_commit();
            tile_regs_wait();
            put(1, cb_e);
            tile_regs_release();
        }
        cb_pop_front(cb_m, 1);
        cb_pop_front(cb_lg, D1);
        // rz = 1 / (e_0 + e_1 + ...)
        cb_wait_front(cb_e, D1);
        tile_regs_acquire();
        load(cb_e, 0, 2);
        for (uint32_t d = 1; d < D1; ++d) {
            load(cb_e, d, 1);
            add_binary_tile_init();
            add_binary_tile(2, 1, 2);
        }
        recip_tile_init();
        recip_tile(2);
        tile_regs_commit();
        tile_regs_wait();
        put(2, cb_rz);
        tile_regs_release();
        // w_d = e_d * rz -> writer (column extraction)
        cb_wait_front(cb_rz, 1);
        for (uint32_t d = 0; d < D1; ++d) {
            tile_regs_acquire();
            load(cb_e, d, 1);
            load(cb_rz, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            tile_regs_commit();
            tile_regs_wait();
            put(1, cb_w);
            tile_regs_release();
        }
        cb_pop_front(cb_rz, 1);
        cb_pop_front(cb_e, D1);
        // B_dq = column 0 of the writer's tile broadcast across the row
        cb_wait_front(cb_wc, D1 * NQ);
        for (uint32_t i = 0; i < D1 * NQ; ++i) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_wc);
            unary_bcast_init<BroadcastType::COL>(cb_wc);
            unary_bcast<BroadcastType::COL>(cb_wc, i, 0);
            unary_bcast_uninit<BroadcastType::COL>(cb_wc);
            tile_regs_commit();
            tile_regs_wait();
            put(0, cb_b);
            tile_regs_release();
        }
        cb_pop_front(cb_wc, D1 * NQ);
        cb_wait_front(cb_b, D1 * NQ);
        // out_q = sum_d n_d * B_dq, per column tile
        for (uint32_t j = 0; j < ntiles; ++j) {
            cb_wait_front(cb_n, D1);
            tile_regs_acquire();
            for (uint32_t d = 0; d < D1; ++d) {
                load(cb_n, d, NQ);
                for (uint32_t q = 0; q < NQ; ++q) {
                    load(cb_b, d * NQ + q, NQ + 1);
                    mul_binary_tile_init();
                    if (d == 0) {
                        mul_binary_tile(NQ, NQ + 1, q);
                    } else {
                        mul_binary_tile(NQ, NQ + 1, NQ + 1);
                        add_binary_tile_init();
                        add_binary_tile(q, NQ + 1, q);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, NQ);
            pack_reconfig_data_format(cb_out);
            for (uint32_t q = 0; q < NQ; ++q) {
                pack_tile(q, cb_out);
            }
            cb_push_back(cb_out, NQ);
            tile_regs_release();
            cb_pop_front(cb_n, D1);
        }
        cb_pop_front(cb_b, D1 * NQ);
    }
}
