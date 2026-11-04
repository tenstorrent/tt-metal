// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DSA index core (kernels/mla/ix_common.hpp) — compute, every SFPU stage in exact fp32 DST (fp32 CBs unpacked
// straight to DST): LayerNorm_128 of k for all T rows at once -> [kn | g] bf16; wsc = w * 2^-6; then per
// active token the pooled key from its slot's updated ring (column softmax over ring rows 0..3, rows 4..31 masked).
// Compile-time args: T_MAX, INV_128_BITS, EPS_BITS, WSCALE_BITS.
// Runtime args: T (+ the lane with MLA_ATT_LANES, rev 47).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/ix_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"

namespace {
using namespace mlaix;
constexpr uint32_t INV_128_BITS = get_compile_time_arg_val(1);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);
constexpr uint32_t WSCALE_BITS = get_compile_time_arg_val(3);
constexpr uint32_t CB_MEAN = CB_T0, CB_D = CB_T0 + 1, CB_INV = CB_T0 + 2, CB_L = CB_T0 + 3, CB_M = CB_T0 + 4,
                   CB_E = CB_T0 + 5;

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, i, d);
}
FORCE_INLINE void emit(uint32_t d, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
    tile_regs_release();
}
FORCE_INLINE void add(uint32_t a, uint32_t b) {
    add_binary_tile_init();
    add_binary_tile(a, b, a);
}
}  // namespace

void kernel_main() {
    const uint32_t t_rows = get_arg_val<uint32_t>(0);
#if defined(MLA_ATT_LANES)  // rev 47: one index core per token lane (tt/mk_mla.py _ix_cores): this lane's tokens only
    const uint32_t t0 = get_arg_val<uint32_t>(1);
#else
    const uint32_t t0 = 0;
#endif
    compute_kernel_hw_startup(CB_Y, CB_ROW);
    cb_wait_front(CB_C, 13);
    cb_wait_front(CB_INFO, 1);
    cb_wait_front(CB_Y, 9);
    // mean over the 128 k columns (column 0 of CB_MEAN)
    cb_reserve_back(CB_MEAN, 1);
    tile_regs_acquire();
    load(CB_Y, 0, 0);
    for (uint32_t i = 1; i < KT; ++i) {
        load(CB_Y, i, 1);
        add(0, 1);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_128_BITS);
    emit(0, CB_MEAN);
    cb_push_back(CB_MEAN, 1);
    cb_wait_front(CB_MEAN, 1);
    // centred k
    cb_reserve_back(CB_D, KT);
    for (uint32_t i = 0; i < KT; ++i) {
        tile_regs_acquire();
        load(CB_Y, i, 0);
        load(CB_MEAN, 0, 1);
        sfpu_sub_bcast_col_init();
        sfpu_sub_bcast_col(0, 1);
        emit(0, CB_D);
    }
    cb_push_back(CB_D, KT);
    cb_wait_front(CB_D, KT);
    // 1 / sqrt(var + eps)
    cb_reserve_back(CB_INV, 1);
    tile_regs_acquire();
    load(CB_D, 0, 0);
    square_tile_init();
    square_tile(0);
    for (uint32_t i = 1; i < KT; ++i) {
        load(CB_D, i, 1);
        square_tile_init();
        square_tile(1);
        add(0, 1);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_128_BITS);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    emit(0, CB_INV);
    cb_push_back(CB_INV, 1);
    cb_wait_front(CB_INV, 1);
    // [kn | g] bf16
    cb_reserve_back(CB_ROW, RING_TILES);
    for (uint32_t i = 0; i < KT; ++i) {
        tile_regs_acquire();
        load(CB_D, i, 0);
        load(CB_INV, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        load(CB_C, i, 1);  // weight, row 0
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        load(CB_C, 4 + i, 1);  // bias, row 0
        sfpu_add_bcast_row_init();
        sfpu_add_bcast_row(0, 1);
        emit(0, CB_ROW);
    }
    for (uint32_t i = 0; i < GT; ++i) {
        tile_regs_acquire();
        load(CB_Y, KT + i, 0);
        emit(0, CB_ROW);
    }
    cb_push_back(CB_ROW, RING_TILES);
    cb_pop_front(CB_D, KT);
    cb_pop_front(CB_INV, 1);
    cb_pop_front(CB_MEAN, 1);
    // wsc: row t = token t's 32 head weights (the in0 of the owners' head-weighted score matmul)
    cb_reserve_back(CB_W, 1);
    tile_regs_acquire();
    load(CB_Y, KT + GT, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, WSCALE_BITS);
    emit(0, CB_W);
    cb_push_back(CB_W, 1);
    cb_pop_front(CB_Y, 9);
    // pooled keys, per active token (of this index core's token lane)
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t p = static_cast<int32_t>(read_tile_value(CB_INFO, 0, t));
        if (p < 0) {
            continue;
        }
        cb_wait_front(CB_RING, RING_TILES);
        cb_reserve_back(CB_L, GT);
        for (uint32_t i = 0; i < GT; ++i) {  // logits = g + ape (rows 0..3), rows 4..31 -> -1e30
            tile_regs_acquire();
            load(CB_RING, KT + i, 0);
            load(CB_C, 8 + i, 1);
            add(0, 1);
            load(CB_C, 12, 1);
            add(0, 1);
            emit(0, CB_L);
        }
        cb_push_back(CB_L, GT);
        cb_wait_front(CB_L, GT);
        cb_reserve_back(CB_M, GT);
        for (uint32_t i = 0; i < GT; ++i) {  // column max (row 0)
            tile_regs_acquire();
            load(CB_L, i, 0);
            sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
            sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
            emit(0, CB_M);
        }
        cb_push_back(CB_M, GT);
        cb_wait_front(CB_M, GT);
        cb_reserve_back(CB_E, GT);
        for (uint32_t i = 0; i < GT; ++i) {  // e = exp(logits - max)
            tile_regs_acquire();
            load(CB_L, i, 0);
            load(CB_M, i, 1);
            sfpu_sub_bcast_row_init();
            sfpu_sub_bcast_row(0, 1);
            exp_tile_init<false>();
            exp_tile<false>(0);
            emit(0, CB_E);
        }
        cb_push_back(CB_E, GT);
        cb_wait_front(CB_E, GT);
        cb_pop_front(CB_L, GT);
        cb_pop_front(CB_M, GT);
        cb_reserve_back(CB_POOL, POOL_TILES);
        for (uint32_t i = 0; i < GT; ++i) {  // pooled = colsum(e k) / colsum(e)
            tile_regs_acquire();
            load(CB_E, i, 1);
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(1);
            load(CB_E, i, 0);
            load(CB_RING, i, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
            div_binary_tile_init();
            div_binary_tile(0, 1, 0);
            emit(0, CB_POOL);
        }
        cb_push_back(CB_POOL, POOL_TILES);
        cb_pop_front(CB_E, GT);
        cb_pop_front(CB_RING, RING_TILES);
    }
    cb_pop_front(CB_INFO, 1);
    cb_pop_front(CB_C, 13);
}
