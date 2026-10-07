// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"

namespace {
// All input copies unpack directly into FP32 DEST. No FPU arithmetic or
// DEST-to-Src reuse is used for either the state or the cancellation in delta.
void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_full_operand<SrcOrder::Regular>(cb, cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
template <BroadcastType B>
void broadcast(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_full_operand<SrcOrder::Regular>(cb, cb);
    unary_bcast_init<B>(cb);
    unary_bcast<B>(cb, tile, dst);
    unary_bcast_uninit<B>(cb);
}
void multiply(uint32_t a, uint32_t b, uint32_t dst) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, dst);
}
void add(uint32_t a, uint32_t b, uint32_t dst) {
    add_binary_tile_init();
    add_binary_tile(a, b, dst);
}
// The caller owns a reserved output window and an acquired DEST bank.
void emit(uint32_t dst, uint32_t cb, uint32_t tile) {
    pack_reconfig_data_format<true>(cb);
    pack_init(cb);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(dst, cb, tile);
    tile_regs_release();
}
// The model's raw convolution vectors are BF16 values represented exactly in
// FP32. Preserve FP32 through squaring, reduction, reciprocal sqrt and scaling:
// the earlier generic normalization sequence failed the error gate on
// cancellation-sensitive real-weight heads. CB13 is compute-private scratch;
// CB11/12 retain the normalized columns for this work item only.
void normalize(uint32_t source, uint32_t target, bool query) {
    cb_reserve_back(13, 1);
    tile_regs_acquire();
    for (uint32_t kr = 0; kr < 4; ++kr) {
        const uint32_t product = kr == 0 ? 3 : 0;
        broadcast<BroadcastType::COL>(source, kr, product);
        multiply(product, product, product);
        if (kr != 0) {
            add(3, 0, 3);
        }
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
    binop_with_scalar_tile_init();
    add_unary_tile(3, 0x358637bd);  // FP32 1e-6, matching the model L2 epsilon.
    rsqrt_tile_init();
    rsqrt_tile(3);
    if (query) {
        binop_with_scalar_tile_init();
        mul_unary_tile(3, 0x3db504f3);  // FP32 1/sqrt(128).
    }
    emit(3, 13, 0);
    cb_push_back(13, 1);
    cb_wait_front(13, 1);
    cb_reserve_back(target, 4);
    for (uint32_t kr = 0; kr < 4; ++kr) {
        tile_regs_acquire();
        broadcast<BroadcastType::COL>(source, kr, 0);
        broadcast<BroadcastType::SCALAR>(13, 0, 1);
        multiply(0, 1, 0);
        emit(0, target, kr);
    }
    cb_push_back(target, 4);
    cb_pop_front(13, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t value_columns = get_compile_time_arg_val(0);
    constexpr bool normalize_qk = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t q_cb = normalize_qk ? 11 : 0;
    constexpr uint32_t k_cb = normalize_qk ? 12 : 1;
    static_assert(value_columns == 1 || value_columns == 2 || value_columns == 4);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 2, 8);
    for (uint32_t item = 0; item < count; ++item) {
        cb_wait_front(0, 4);
        cb_wait_front(1, 4);
        cb_wait_front(2, value_columns);
        cb_wait_front(3, 1);
        cb_wait_front(4, 1);
        cb_wait_front(5, 4 * value_columns);
        if constexpr (normalize_qk) {
            normalize(0, q_cb, true);
            normalize(1, k_cb, false);
            cb_wait_front(q_cb, 4);
            cb_wait_front(k_cb, 4);
        }
        cb_reserve_back(6, value_columns);
        // delta = beta * (v - k^T (decay * S)). Four partial rows
        // accumulate in DEST before the FP32 column reduction over K.
        for (uint32_t vc = 0; vc < value_columns; ++vc) {
            tile_regs_acquire();
            for (uint32_t kr = 0; kr < 4; ++kr) {
                const uint32_t product = kr == 0 ? 3 : 0;
                load(5, kr * value_columns + vc, product);
                broadcast<BroadcastType::SCALAR>(3, 0, 1);
                multiply(product, 1, product);
                broadcast<BroadcastType::COL>(k_cb, kr, 1);
                multiply(product, 1, product);
                if (kr != 0) {
                    add(3, 0, 3);
                }
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
            load(2, vc, 1);
            sub_binary_tile_init();
            sub_binary_tile(1, 3, 0);
            broadcast<BroadcastType::SCALAR>(4, 0, 1);
            multiply(0, 1, 0);
            emit(0, 6, vc);
        }
        cb_push_back(6, value_columns);
        cb_wait_front(6, value_columns);
        cb_reserve_back(7, 4 * value_columns);
        // S_new = decay * S + k outer delta, preserving FP32 throughout.
        for (uint32_t kr = 0; kr < 4; ++kr) {
            for (uint32_t vc = 0; vc < value_columns; ++vc) {
                tile_regs_acquire();
                load(5, kr * value_columns + vc, 0);
                broadcast<BroadcastType::SCALAR>(3, 0, 1);
                multiply(0, 1, 0);
                broadcast<BroadcastType::COL>(k_cb, kr, 1);
                broadcast<BroadcastType::ROW>(6, vc, 2);
                multiply(1, 2, 1);
                add(0, 1, 0);
                emit(0, 7, kr * value_columns + vc);
            }
        }
        cb_push_back(7, 4 * value_columns);
        cb_wait_front(7, 4 * value_columns);
        cb_reserve_back(8, value_columns);
        // Output = q^T S_new. The writer may read CB7 concurrently for its
        // DRAM write, but waits for CB8 BEFORE reclaiming CB7's pages.
        for (uint32_t vc = 0; vc < value_columns; ++vc) {
            tile_regs_acquire();
            for (uint32_t kr = 0; kr < 4; ++kr) {
                const uint32_t product = kr == 0 ? 3 : 0;
                load(7, kr * value_columns + vc, product);
                broadcast<BroadcastType::COL>(q_cb, kr, 1);
                multiply(product, 1, product);
                if (kr != 0) {
                    add(3, 0, 3);
                }
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
            emit(3, 8, vc);
        }
        cb_push_back(8, value_columns);
        cb_pop_front(0, 4);
        cb_pop_front(1, 4);
        cb_pop_front(2, value_columns);
        cb_pop_front(3, 1);
        cb_pop_front(4, 1);
        cb_pop_front(5, 4 * value_columns);
        cb_pop_front(6, value_columns);
        if constexpr (normalize_qk) {
            cb_pop_front(q_cb, 4);
            cb_pop_front(k_cb, 4);
        }
        // CB7 and CB8 have exactly one consumer: the writer.
    }
}
