// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the fused V4.1 mHC stream mix: per unit, out_j = in_0 * coef[j][0] + sum_{k >= 1} in_k * coef[j][k],
// all fp32 on the SFPU (inputs in fp32 DST), the terms accumulated in order k = 0, 1, ...: the same
// multiply-then-addcmul sequence, in the same order, as the composite ttnn path it replaces. The fp32 result is
// packed to cb_out's format: fp32, or (OUT_BF16, a collapse feeding a bf16 sublayer) rounded to nearest-even on
// the SFPU first, as ttnn.typecast does (the packer alone rounds ties differently).
//
// Inputs: the streams (cb_in, fp32, unpacked straight to DST) and, with HAS_X, the extra term 0 from cb_x (fp32
// or bf16; a bf16 tile unpacks exactly into fp32 DST through SrcA). Work arrives in blocks of BLOCK units of one
// tile row (same coefficients). Per output j the block's BLOCK accumulators live in DST together, so each
// coefficient tile is unpacked once per block rather than once per unit. Needs dst_full_sync_en (8 fp32 DST
// tiles: BLOCK accumulators, one input slot, one coefficient slot).
//
// compile_time_args = [cb_in, cb_coef, cb_out, CT, N, J, BLOCK, HAS_X, cb_x, X_FP32, OUT_BF16]
// runtime args      = [unit_start, unit_count]  (unit_count a multiple of BLOCK)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t CT = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr uint32_t J = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(6);
    constexpr uint32_t HAS_X = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(8);
    constexpr bool X_FP32 = get_compile_time_arg_val(9) == 1;
    constexpr bool OUT_BF16 = get_compile_time_arg_val(10) == 1;
    constexpr auto FP32 = static_cast<uint32_t>(DataFormat::Float32);
    constexpr auto BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
    constexpr uint32_t K = N + HAS_X;
    constexpr uint32_t ONE = 0x3F800000u;  // fp32 1.0: addcmul's scalar
    constexpr uint32_t IN = BLOCK;         // DST slots: accumulators 0..BLOCK-1, then the input, then the coefficient
    constexpr uint32_t COEF = BLOCK + 1;
    static_assert(BLOCK + 2 <= 8, "BLOCK accumulators + input + coefficient must fit the 8 fp32 DST tiles");

    if (unit_count == 0) {
        return;
    }

    DataflowBuffer in(cb_in);
    DataflowBuffer coef(cb_coef);
    DataflowBuffer xin(cb_x);
    DataflowBuffer out(cb_out);
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);  // cb_in and cb_coef share the fp32 unpack-to-dest config; SFPU inits leave it alone

    // term k of unit b: the extra input (cb_x tile b) first, then stream i = k - HAS_X (cb_in tile b * N + i)
    auto copy_term = [&](uint32_t k, uint32_t b) {
        if (HAS_X && k == 0) {
            copy_tile(cb_x, b, IN);
        } else {
            copy_tile(cb_in, b * N + k - HAS_X, IN);
        }
    };

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
        in.wait_front(BLOCK * N);  // unit-major: tile b * N + i
        if constexpr (HAS_X) {
            xin.wait_front(BLOCK);
        }
        out.reserve_back(BLOCK * J);  // unit-major: tile b * J + j
        for (uint32_t j = 0; j < J; ++j) {
            tile_regs_acquire();
            copy_tile(cb_coef, j * K, COEF);
            if constexpr (HAS_X && !X_FP32) {
                // the bf16 term goes through SrcA: switch the unpacker to its format, and back after the term
                reconfig_data_format_srca(cb_coef, cb_x);
                copy_init(cb_x);
            }
            mul_binary_tile_init();
            for (uint32_t b = 0; b < BLOCK; ++b) {
                copy_term(0, b);
                mul_binary_tile(IN, COEF, b);
            }
            if constexpr (HAS_X && !X_FP32) {
                reconfig_data_format_srca(cb_x, cb_in);
                copy_init(cb_in);
            }
            addcmul_tile_init();
            for (uint32_t k = 1; k < K; ++k) {
                copy_tile(cb_coef, j * K + k, COEF);
                for (uint32_t b = 0; b < BLOCK; ++b) {
                    copy_term(k, b);
                    addcmul_tile<DataFormat::Float32>(b, IN, COEF, b, ONE);
                }
            }
            if constexpr (OUT_BF16) {
                typecast_tile_init<FP32, BF16>();
                for (uint32_t b = 0; b < BLOCK; ++b) {
                    typecast_tile<FP32, BF16>(b);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < BLOCK; ++b) {
                pack_tile<true>(b, cb_out, b * J + j);
            }
            tile_regs_release();
        }
        in.pop_front(BLOCK * N);
        if constexpr (HAS_X) {
            xin.pop_front(BLOCK);
        }
        out.push_back(BLOCK * J);
    }
    coef.pop_front(J * K);
}
