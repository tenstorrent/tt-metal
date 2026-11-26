// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill causal conv, compute. Per output tile (R, j):
//   lag r rows: L_r = A_r x[R] + B_r x[R - 1] (0/1 shift matrices, HiFi4 matmul: bf16 x 1 sums one term, exact)
//   t_r = tap row r broadcast down the tile; acc = L_0 * t_0 + L_1 * t_1 + L_2 * t_2 + L_3 * t_3   (fp32 SFPU,
//   prefill_model._conv's order) mode 0: bf16(acc); mode 1: fp32(bf16(acc)); mode 2: bf16(resid + fp32(bf16(acc)));
//   mode 3: fp32(resid) + fp32(bf16(acc)), fp32 out (resid bf16 or fp32)
//   mode 4 (dyn): t_r = D_r + tap row r, D_r = dyn tile x selection tile r (0/1 matmul: dyn column 4 h + r broadcast
//   over the columns, exact); out = bf16((acc + L_0) * scale tile)   (Inkling2Prefill._kv's global order)
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void tap(uint32_t cb, uint32_t tile, uint32_t dst) {  // row 0 broadcast down the tile (fp32 exact)
    reconfig_data_format_srca(cb);
    unary_bcast_init<BroadcastType::ROW>(cb);
    unary_bcast<BroadcastType::ROW>(cb, tile, dst);
    unary_bcast_uninit<BroadcastType::ROW>(cb);
}
FORCE_INLINE void to_bf16(uint32_t dst) {  // round to nearest even in dest, as ttnn.typecast (the packer truncates)
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr bool dyn_mode = mode == 4;
    constexpr uint32_t cb_x = 0, cb_prev = 1, cb_taps = 2, cb_sh = 3, cb_res = 4, cb_out = 16, cb_y = 17;
    constexpr uint32_t cb_sel = 5, cb_dyn = 6, cb_sc = 7, cb_dx = 8;
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sh, cb_x, cb_out);
    cb_wait_front(cb_sh, 6);
    if constexpr (dyn_mode) {
        cb_wait_front(cb_sc, 1);
    }
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t rows = get_arg_val<uint32_t>(1 + u);
        cb_wait_front(cb_taps, 4);
        if constexpr (dyn_mode) {
            cb_wait_front(cb_sel, 4);
        }
        for (uint32_t i = 0; i < rows; ++i) {
            if constexpr (dyn_mode) {  // D_0..D_3 -> cb_dx (fp32, exact)
                cb_wait_front(cb_dyn, 1);
                tile_regs_acquire();
                reconfig_data_format(cb_sel, cb_dyn);
                matmul_init(cb_dyn, cb_sel);
                for (uint32_t r = 0; r < 4; ++r) {
                    matmul_tiles(cb_dyn, cb_sel, 0, r, r);
                }
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_dx, 4);
                pack_reconfig_data_format(cb_dx);
                for (uint32_t r = 0; r < 4; ++r) {
                    pack_tile(r, cb_dx);
                }
                cb_push_back(cb_dx, 4);
                tile_regs_release();
                cb_pop_front(cb_dyn, 1);
                cb_wait_front(cb_dx, 4);
            }
            cb_wait_front(cb_x, 1);
            cb_wait_front(cb_prev, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_x, cb_sh);  // matmul: in1 -> SrcA, in0 -> SrcB
            matmul_init(cb_sh, cb_x);
            for (uint32_t r = 1; r < 4; ++r) {
                matmul_tiles(cb_sh, cb_x, r - 1, 0, 2 + r);
            }
            reconfig_data_format(cb_prev, cb_sh);
            matmul_init(cb_sh, cb_prev);
            for (uint32_t r = 1; r < 4; ++r) {
                matmul_tiles(cb_sh, cb_prev, 2 + r, 0, 2 + r);
            }
            load(cb_x, 0, 1);
            tap(cb_taps, 0, 2);
            if constexpr (dyn_mode) {
                load(cb_dx, 0, 6);
                add_binary_tile_init();
                add_binary_tile(6, 2, 2);
            }
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 0);
            for (uint32_t r = 1; r < 4; ++r) {
                tap(cb_taps, r, 2);
                if constexpr (dyn_mode) {
                    load(cb_dx, r, 6);
                    add_binary_tile_init();
                    add_binary_tile(6, 2, 2);
                }
                mul_binary_tile_init();
                mul_binary_tile(2 + r, 2, 2 + r);
                add_binary_tile_init();
                add_binary_tile(0, 2 + r, 0);
            }
            if constexpr (dyn_mode) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                load(cb_sc, 0, 2);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
            }
            to_bf16(0);
            tile_regs_commit();
            tile_regs_wait();
            put(0, (mode == 0 || dyn_mode) ? cb_out : cb_y);
            tile_regs_release();
            cb_pop_front(cb_x, 1);
            cb_pop_front(cb_prev, 1);
            if constexpr (dyn_mode) {
                cb_pop_front(cb_dx, 4);
            }
            if constexpr (mode != 0 && !dyn_mode) {
                cb_wait_front(cb_y, 1);
                tile_regs_acquire();
                load(cb_y, 0, 0);
                if constexpr (mode >= 2) {
                    cb_wait_front(cb_res, 1);
                    load(cb_res, 0, 1);
                    add_binary_tile_init();
                    add_binary_tile(1, 0, 1);
                    if constexpr (mode == 2) {
                        to_bf16(1);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                put(mode >= 2 ? 1 : 0, cb_out);
                tile_regs_release();
                cb_pop_front(cb_y, 1);
                if constexpr (mode >= 2) {
                    cb_pop_front(cb_res, 1);
                }
            }
        }
        cb_pop_front(cb_taps, 4);
        if constexpr (dyn_mode) {
            cb_pop_front(cb_sel, 4);
        }
    }
    cb_pop_front(cb_sh, 6);
    if constexpr (dyn_mode) {
        cb_pop_front(cb_sc, 1);
    }
}
