// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the fused V4.1 mHC stream mix: per unit, out_j = in_0 * coef[j][0] + sum_{k >= 1} in_k * coef[j][k],
// all fp32 on the SFPU (inputs unpacked straight to fp32 DST), the terms accumulated in order k = 0, 1, ...:
// the same multiply-then-addcmul sequence, in the same order, as the composite ttnn path it replaces.
//
// Work arrives in blocks of BLOCK units of one tile row (same coefficients). Per output j the block's BLOCK
// accumulators live in DST together, so each coefficient tile is unpacked once per block rather than once per
// unit. Needs dst_full_sync_en (8 fp32 DST tiles: BLOCK accumulators, one input slot, one coefficient slot).
//
// compile_time_args = [cb_in, cb_coef, cb_out, CT, K, J, BLOCK]
// runtime args      = [unit_start, unit_count]  (unit_count a multiple of BLOCK)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t CT = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t J = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(6);
    constexpr uint32_t ONE = 0x3F800000u;  // fp32 1.0: addcmul's scalar
    constexpr uint32_t IN = BLOCK;         // DST slots: accumulators 0..BLOCK-1, then the input, then the coefficient
    constexpr uint32_t COEF = BLOCK + 1;
    static_assert(BLOCK + 2 <= 8, "BLOCK accumulators + input + coefficient must fit the 8 fp32 DST tiles");

    if (unit_count == 0) {
        return;
    }

    DataflowBuffer in(cb_in);
    DataflowBuffer coef(cb_coef);
    DataflowBuffer out(cb_out);
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);  // cb_in and cb_coef share the fp32 unpack-to-dest config; SFPU inits leave it alone

    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t u = unit_start; u < unit_start + unit_count; u += BLOCK) {
        const uint32_t r = u / CT;
        if (r != row) {
            if (row != 0xFFFFFFFFu) {
                coef.pop_front(J * K);
            }
            row = r;
            coef.wait_front(J * K);
        }
        in.wait_front(BLOCK * K);     // unit-major: tile b * K + k
        out.reserve_back(BLOCK * J);  // unit-major: tile b * J + j
        for (uint32_t j = 0; j < J; ++j) {
            tile_regs_acquire();
            copy_tile(cb_coef, j * K, COEF);
            mul_binary_tile_init();
            for (uint32_t b = 0; b < BLOCK; ++b) {
                copy_tile(cb_in, b * K, IN);
                mul_binary_tile(IN, COEF, b);
            }
            addcmul_tile_init();
            for (uint32_t k = 1; k < K; ++k) {
                copy_tile(cb_coef, j * K + k, COEF);
                for (uint32_t b = 0; b < BLOCK; ++b) {
                    copy_tile(cb_in, b * K + k, IN);
                    addcmul_tile<DataFormat::Float32>(b, IN, COEF, b, ONE);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < BLOCK; ++b) {
                pack_tile<true>(b, cb_out, b * J + j);
            }
            tile_regs_release();
        }
        in.pop_front(BLOCK * K);
        out.push_back(BLOCK * J);
    }
    coef.pop_front(J * K);
}
