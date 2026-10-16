// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single-tile primitive checks for the fused Sol-Attn pre-pass (test_sol_prepass_fused: test_sol_primitives).
// Inputs: one tile A in c_0, one tile B in c_1 (reader_inputs.cpp, NumInputs = 2); output one tile in c_16.
// Compile-time arg 0 = OP:
//   0: C = A @ B                       1: C = A @ T(B) (matmul_init transpose = 1)
//   2: C = gtz(A - B)  (SFPU binary sub + gtz on DST 0/1)
//   3: C = A * s1 + sqrt(B * s1 + s2) (binop-with-scalar, sqrt, SFPU add; s1, s2 = ct args 1, 2 as fp32 bits)
//   4: C = A + B computed in DST tiles 6 and 7 (high DST indices under full sync)
//   5: C = relu(A * A - B)  (mul_binary, sub_binary, relu)
//   6: C = A * s1 computed in DST tile 5 (unary SFPU op on a high DST index)   7: same in DST tile 1
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t OP = get_compile_time_arg_val(0);
    constexpr uint32_t S1 = get_compile_time_arg_val(1);
    constexpr uint32_t S2 = get_compile_time_arg_val(2);
    constexpr uint32_t ca = 0, cb = 1, co = 16;
    CircularBuffer a(ca), b(cb), o(co);
    compute_kernel_hw_startup(ca, cb, co);
    a.wait_front(1);
    b.wait_front(1);
    o.reserve_back(1);
    tile_regs_acquire();
    uint32_t out_dst = 0;
    if constexpr (OP == 0 || OP == 1) {
        matmul_init(ca, cb, OP);
        matmul_tiles(ca, cb, 0, 0, 0);
    } else if constexpr (OP == 2) {
        copy_init(ca);
        copy_tile(ca, 0, 0);
        copy_init(cb);
        copy_tile(cb, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        gtz_tile_init();
        gtz_tile(0);
    } else if constexpr (OP == 3) {
        copy_init(ca);
        copy_tile(ca, 0, 0);
        copy_init(cb);
        copy_tile(cb, 0, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, S1);
        mul_unary_tile(1, S1);
        add_unary_tile(1, S2);
        sqrt_tile_init();
        sqrt_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else if constexpr (OP == 4) {
        copy_init(ca);
        copy_tile(ca, 0, 6);
        copy_init(cb);
        copy_tile(cb, 0, 7);
        add_binary_tile_init();
        add_binary_tile(6, 7, 6);
        out_dst = 6;
    } else if constexpr (OP == 6 || OP == 7) {
        out_dst = OP == 6 ? 5 : 1;
        copy_init(ca);
        copy_tile(ca, 0, out_dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(out_dst, S1);
    } else {
        copy_init(ca);
        copy_tile(ca, 0, 0);
        copy_init(cb);
        copy_tile(cb, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 0);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        relu_tile_init();
        relu_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(out_dst, co, 0);
    tile_regs_release();
    o.push_back(1);
    a.pop_front(1);
    b.pop_front(1);
}
