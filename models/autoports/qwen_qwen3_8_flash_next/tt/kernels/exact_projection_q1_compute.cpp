// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
template <int Iterations>
ALWI void multiply_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, Iterations, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}
template <int Iterations>
ALWI void add_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, Iterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    static_assert(rows == 2 || rows == 4 || rows == 8 || rows == 16);
    // SFPLOAD covers four physical rows x eight columns. Two iterations
    // cover both column halves of that four-row band in each top face.
    // R2 therefore uses the R4 arithmetic footprint; weight rows2/3 are zero.
    // A one-iteration R2 mask would silently omit columns8..15 in each face.
    constexpr uint32_t iterations = rows <= 4 ? 2 : rows / 2;
    const uint32_t first = get_arg_val<uint32_t>(0), stride = get_arg_val<uint32_t>(1);
    const uint32_t groups = get_arg_val<uint32_t>(2), kt = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup(2, 3, 4);
    for (uint32_t g = first; g < groups; g += stride) {
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(2, 0.0f);
        for (uint32_t k = 0; k < kt; ++k) {
            cb_wait_front(2, 1);
            cb_wait_front(3, 1);
            // BF16 ROW broadcast consumes SrcB; FP32 ROW broadcast takes
            // unpack-to-DST via SrcA. Startup configured SrcB from weight CB3,
            // so query broadcast must restore BOTH formats before its init.
            reconfig_data_format(2, 2);
            unary_bcast_init<BroadcastType::ROW>(2);
            unary_bcast<BroadcastType::ROW>(2, 0, 0);
            unary_bcast_uninit<BroadcastType::ROW>(2);
            reconfig_data_format_srca(3);
            copy_init(3);
            copy_tile(3, 0, 1);
            mul_binary_tile_init();
            multiply_rows<iterations>(0, 1, k == 0 ? 2 : 0);
            if (k != 0) {
                add_binary_tile_init();
                add_rows<iterations>(2, 0, 2);
            }
            cb_pop_front(2, 1);
            cb_pop_front(3, 1);
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
        cb_reserve_back(4, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, 4);
        tile_regs_release();
        cb_push_back(4, 1);
    }
}
