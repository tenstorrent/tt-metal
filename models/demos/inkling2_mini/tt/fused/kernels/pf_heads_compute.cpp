// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill per-head norm, compute. Per unit (tile row R, head h), over the head's 4 column tiles c:
//   y_c = x_c [* scale_c (row 0 broadcast down)]          fp32 SFPU
//   q = sum_c y_c^2 (fp32 SFPU), hi = bf16(q), lo = q - hi; s = row sums of hi + row sums of lo (reduce row, column 0
//   of the f tile: the FPU reads both parts exactly, a plain fp32 reduce would truncate q)
//   norm 0 (L2 clip): f = rsqrt(max(s * inv, 1));  norm 1 (RMS): f = rsqrt(s * inv + eps)
//   out_c = bf16(y_c * f) (f: column 0 broadcast across), then [bf16(out_c * tau)] [bf16(out_c * mult)]: the op
//   chain's (and the bf16 reference model's) rounding points
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_x = 0, cb_scale = 1, cb_tau = 2, cb_scaler = 3;
constexpr uint32_t cb_y = 16, cb_sq = 17, cb_f = 18, cb_out = 19, cb_hl = 20;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
template <BroadcastType B>
FORCE_INLINE void bcast(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    unary_bcast_init<B>(cb);
    unary_bcast<B>(cb, tile, dst);
    unary_bcast_uninit<B>(cb);
}
FORCE_INLINE void to_bf16(uint32_t dst) {  // round to nearest even in dest, as ttnn.typecast
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
    constexpr uint32_t has_scale = get_named_compile_time_arg_val("has_scale");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t norm = get_named_compile_time_arg_val("norm");
    constexpr uint32_t inv = get_named_compile_time_arg_val("inv");    // fp32 bits
    constexpr uint32_t eps = get_named_compile_time_arg_val("eps");    // fp32 bits
    constexpr uint32_t mult = get_named_compile_time_arg_val("mult");  // fp32 bits
    constexpr uint32_t one = 0x3f800000u;
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_y);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t u = 0; u < n_units; ++u) {
        cb_wait_front(cb_x, 4);
        if constexpr (has_scale) {
            cb_wait_front(cb_scale, 4);
        }
        for (uint32_t c = 0; c < 4; ++c) {
            tile_regs_acquire();
            load(cb_x, c, 0);
            if constexpr (has_scale) {
                bcast<BroadcastType::ROW>(cb_scale, c, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
            }
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 1);
            tile_regs_commit();
            tile_regs_wait();
            put(0, cb_y);
            put(1, cb_sq);
            tile_regs_release();
        }
        cb_pop_front(cb_x, 4);
        if constexpr (has_scale) {
            cb_pop_front(cb_scale, 4);
        }

        cb_wait_front(cb_sq, 4);
        tile_regs_acquire();
        load(cb_sq, 0, 0);
        for (uint32_t c = 1; c < 4; ++c) {
            load(cb_sq, c, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);  // lo = q - hi
        tile_regs_commit();
        tile_regs_wait();
        put(1, cb_hl);
        put(0, cb_hl);
        tile_regs_release();
        cb_pop_front(cb_sq, 4);

        cb_wait_front(cb_hl, 2);
        tile_regs_acquire();
        reconfig_data_format(cb_hl, cb_scaler);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_hl, cb_scaler, cb_f);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_hl, cb_scaler, 0, 0, 0);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_hl, cb_scaler, 1, 0, 0);
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv);
        if constexpr (norm == 1) {
            add_unary_tile(0, eps);
        } else {
            unary_max_tile_init();
            unary_max_tile(0, one);
        }
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_f);
        tile_regs_release();
        cb_pop_front(cb_hl, 2);

        cb_wait_front(cb_y, 4);
        cb_wait_front(cb_f, 1);
        if constexpr (has_tau) {
            cb_wait_front(cb_tau, 1);
        }
        for (uint32_t c = 0; c < 4; ++c) {
            tile_regs_acquire();
            load(cb_y, c, 0);
            bcast<BroadcastType::COL>(cb_f, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            to_bf16(0);
            if constexpr (has_tau) {  // the op chain's roundings: bf16 after the clip, after tau, after mult
                bcast<BroadcastType::COL>(cb_tau, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                to_bf16(0);
            }
            if constexpr (mult != one) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, mult);
                to_bf16(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            put(0, cb_out);
            tile_regs_release();
        }
        cb_pop_front(cb_y, 4);
        cb_pop_front(cb_f, 1);
        if constexpr (has_tau) {
            cb_pop_front(cb_tau, 1);
        }
    }
    cb_pop_front(cb_scaler, 1);
}
