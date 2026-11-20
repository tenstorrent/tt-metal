// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Finisher of the Hy4 decode MoE program (TRISC) (fused/moe/program.py): for each of this core's units (union slot,
// pair of gate / up tile columns) sum the S FP32 partial sums the gate|up cores delivered, apply the routed expert's
// activation and the rows' routing weights:
//     act = silu(min(gate, limit)) * clamp(up, -limit, limit) * w[row]
// packed as one BF16 [ROWS, 32] tile (the down cores' in0). The units of the slots from first_plain on (the shared
// expert, the dense FFN) are plain SwiGLU: no clamp.
//
// cb_part: entry n = unit n's S partials, each [gate tile, up tile] (FP32). cb_sc: one FP32 tile per slot, row r
// filled with w[r][slot] (the writer builds it).
//
// This core finishes the slots f, f + NF, ..., P units each; f is its sync word CFG_INDEX.
// compile-time args: [cb_part, cb_sc, cb_act, S, limit, -limit, -big (float bits), P, NF, k_in, k_act (float bits)]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "moe_common.hpp"

#ifndef FIN_DEBUG
#define FIN_DEBUG 0
#endif

constexpr uint32_t kRowIters = ROWS <= 4 ? 2 : ROWS / 2;
ALWI void clamp_rows(uint32_t idst, uint32_t lo, uint32_t hi) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_clamp, (APPROX, kRowIters), idst, VectorMode::R, lo, hi));
}
ALWI void silu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, kRowIters), idst, VectorMode::R));
}
ALWI void scale_rows(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, kRowIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
ALWI void mul_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

void kernel_main() {
#ifdef HY4_LAUNCH_ONLY  // measurement only: what launching this program costs
    return;
#endif
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t limit = get_compile_time_arg_val(4);
    constexpr uint32_t neg_limit = get_compile_time_arg_val(5);
    constexpr uint32_t neg_big = get_compile_time_arg_val(6);
    constexpr uint32_t P = get_compile_time_arg_val(7);
    constexpr uint32_t NF = get_compile_time_arg_val(8);
    // Truncation bias corrections (float bits; 1.0 = off): k_in multiplies the gate and up sums (the LoFi matmul
    // drops x's last bit and the partials arrive as truncated BF16), k_act the activation (packed as truncated BF16,
    // and the down matmul drops its last bit).
    constexpr uint32_t k_in = get_compile_time_arg_val(9);
    constexpr uint32_t k_act = get_compile_time_arg_val(10);
    constexpr uint32_t one = 0x3F800000;

    compute_kernel_hw_startup(cb_part, cb_sc, cb_act);

    const uint32_t f = hy4moe::cfg(hy4moe::CFG_INDEX);
    uint32_t n_units = 0;
    uint32_t first_plain = 0;  // slots from here on are plain SwiGLU (no clamp)
    UNPACK(n_units = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(n_units = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(n_units = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    UNPACK(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    for (uint32_t n = 0; n < n_units; ++n) {
        const uint32_t base = n * 2 * S;
        cb_wait_front(cb_part, base + 2 * S);  // the partial buffer is filled in unit order and never popped
        cb_wait_front(cb_sc, 1);
        tile_regs_acquire();
        if constexpr (S == 1) {
            copy_init(cb_part);
            copy_tile(cb_part, base, 0);
            copy_tile(cb_part, base + 1, 1);
        } else {
            add_init(cb_part, cb_part);
            add_tiles(cb_part, cb_part, base, base + 2, 0);
            add_tiles(cb_part, cb_part, base + 1, base + 3, 1);
            if constexpr (S > 2) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_part);
                for (uint32_t q = 2; q < S; ++q) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_part, base + 2 * q, 0);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_part, base + 2 * q + 1, 1);
                }
            }
        }
        copy_init(cb_sc);
        copy_tile(cb_sc, 0, 2);

#if FIN_DEBUG == 1  // measurement only: the activation tile is the gate sum (times the row weight)
        mul_binary_tile_init();
        mul_rows(0, 2, 0);
#elif FIN_DEBUG == 2  // measurement only: the up sum
        mul_binary_tile_init();
        mul_rows(1, 2, 0);
#elif FIN_DEBUG == 3  // measurement only: silu(gate)
        silu_tile_init();
        silu_rows(0);
        mul_binary_tile_init();
        mul_rows(0, 2, 0);
#else
        if constexpr (k_in != one) {
            binop_with_scalar_tile_init();
            scale_rows(0, k_in);
            scale_rows(1, k_in);
        }
        if (f + (n / P) * NF < first_plain) {  // unit n is pair n % P of slot f + (n / P) * NF
            clamp_tile_init();
            clamp_rows(0, neg_big, limit);
            clamp_rows(1, neg_limit, limit);
        }
        silu_tile_init();
        silu_rows(0);
        mul_binary_tile_init();
        mul_rows(0, 1, 0);
        mul_rows(0, 2, 0);
#endif
        if constexpr (k_act != one) {
            binop_with_scalar_tile_init();
            scale_rows(0, k_act);
        }
        tile_regs_commit();

        cb_reserve_back(cb_act, 1);
        tile_regs_wait();
        pack_tile(0, cb_act);
        tile_regs_release();
        cb_push_back(cb_act, 1);
        if (n % P == P - 1) {  // the slot's scale tile served its P units
            cb_pop_front(cb_sc, 1);
        }
    }
}
