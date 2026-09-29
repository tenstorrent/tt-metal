// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the fused V4.1 mHC projection: one pass over the fp32 streams x computes, per tile row,
//   out = x @ w                                    (the unnormalized mixes; w is zero from column MIX_COL on)
//   out[:, MIX_COL] = sum(x^2) * INV_WIDTH + EPS    (this chip's share of mean(x^2) + eps over the full width)
// The matmul is the composite path's (tf32 operands, fp32 DST accumulation). The squares are accumulated
// elementwise in fp32 on the SFPU (x unpacked straight to fp32 DST) into a [32, 32] partial-sum tile S, which is
// scaled (+ EPS / 32 per entry) and row-summed into column MIX_COL by a matmul with E (ones in column MIX_COL).
// So the matmul's tf32 operand rounding does not bias the sum, S goes through it as S_hi (S with the low 13
// mantissa bits cleared: exact in tf32) plus S_lo = S - S_hi. The mixes tile is reloaded exactly (fp32 unpack to
// DST) underneath those matmuls.
//
// compile_time_args = [cb_x, cb_xsq, cb_w, cb_const, cb_mix, cb_hi, cb_lo, cb_out, KT, BK,
//                      inv_width_bits, eps_entry_bits]
// runtime args      = [row_count]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t row_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xsq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_const = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mix = get_compile_time_arg_val(4);
    constexpr uint32_t cb_hi = get_compile_time_arg_val(5);
    constexpr uint32_t cb_lo = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t KT = get_compile_time_arg_val(8);
    constexpr uint32_t BK = get_compile_time_arg_val(9);
    constexpr uint32_t INV_WIDTH = get_compile_time_arg_val(10);
    constexpr uint32_t EPS_ENTRY = get_compile_time_arg_val(11);
    constexpr uint32_t NB = KT / BK;
    constexpr uint32_t E_TILE = 0;         // cb_const holds E
    constexpr uint32_t ONE = 0x3F800000u;  // fp32 1.0: addcmul's scalar
    constexpr uint32_t TF32_MASK = 0xFFFFE000u;
    constexpr uint32_t MIX = 0, SQ = 1, IN = 2;  // DST slots

    if (row_count == 0) {
        return;
    }

    DataflowBuffer x(cb_x);
    DataflowBuffer xsq(cb_xsq);
    DataflowBuffer w(cb_w);
    DataflowBuffer consts(cb_const);
    DataflowBuffer mix(cb_mix);
    DataflowBuffer hi(cb_hi);
    DataflowBuffer lo(cb_lo);
    DataflowBuffer out(cb_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    consts.wait_front(1);

    for (uint32_t r = 0; r < row_count; ++r) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < NB; ++b) {
            x.wait_front(BK);
            w.wait_front(BK);
            matmul_init(cb_x, cb_w, 0);
            for (uint32_t i = 0; i < BK; ++i) {
                matmul_tiles(cb_x, cb_w, i, i, MIX);  // DST[MIX] += x_k @ w_k
            }
            xsq.wait_front(BK);
            copy_init(cb_xsq);
            if (b == 0) {
                copy_tile(cb_xsq, 0, IN);
                mul_binary_tile_init();
                mul_binary_tile(IN, IN, SQ);
            }
            addcmul_tile_init();
            for (uint32_t i = (b == 0); i < BK; ++i) {
                copy_tile(cb_xsq, i, IN);
                addcmul_tile<DataFormat::Float32>(SQ, IN, IN, SQ, ONE);  // DST[SQ] += x_k * x_k
            }
            x.pop_front(BK);
            xsq.pop_front(BK);
            w.pop_front(BK);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(SQ, INV_WIDTH);
        add_unary_tile(SQ, EPS_ENTRY);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(SQ, IN);
        bitwise_and_tile_init();
        bitwise_and_tile<DataFormat::Int32>(IN, TF32_MASK);  // S_hi
        sub_binary_tile_init();
        sub_binary_tile(SQ, IN, SQ);  // S_lo = S - S_hi
        tile_regs_commit();
        tile_regs_wait();
        mix.reserve_back(1);
        hi.reserve_back(1);
        lo.reserve_back(1);
        pack_tile(MIX, cb_mix);
        pack_tile(IN, cb_hi);
        pack_tile(SQ, cb_lo);
        mix.push_back(1);
        hi.push_back(1);
        lo.push_back(1);
        tile_regs_release();

        mix.wait_front(1);
        hi.wait_front(1);
        lo.wait_front(1);
        tile_regs_acquire();
        copy_init(cb_mix);
        copy_tile(cb_mix, 0, MIX);
        matmul_init(cb_hi, cb_const, 0);
        matmul_tiles(cb_hi, cb_const, 0, E_TILE, MIX);  // DST[MIX][:, MIX_COL] += rowsum(S_hi)
        matmul_init(cb_lo, cb_const, 0);
        matmul_tiles(cb_lo, cb_const, 0, E_TILE, MIX);  // ... + rowsum(S_lo)
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(MIX, cb_out);
        out.push_back(1);
        tile_regs_release();
        mix.pop_front(1);
        hi.pop_front(1);
        lo.pop_front(1);
    }
    consts.pop_front(1);
}
