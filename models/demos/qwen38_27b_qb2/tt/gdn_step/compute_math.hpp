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
