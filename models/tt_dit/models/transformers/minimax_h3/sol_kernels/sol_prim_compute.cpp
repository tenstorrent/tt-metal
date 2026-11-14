// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single-tile checks of the compute primitives the Sol-Attn pre-pass uses (test_sol_primitives).
// Inputs: tile A in c_0, tile B in c_1; output tile in c_16. Compile-time arg 0 = OP:
//   0: A @ B                 1: A @ B^T
//   2: gtz(A - B)            3: A*s1 + sqrt(B*s1 + s2)   (s1, s2 = SCALAR_1, SCALAR_2: args 1, 2 as fp32 bits)
//   4: A + B in DST tiles 6, 7
//   5: relu(A*A - B)
//   6: A*s1 in DST tile 5    7: A*s1 in DST tile 1

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
    constexpr uint32_t SCALAR_1 = get_compile_time_arg_val(1);
    constexpr uint32_t SCALAR_2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;
    CircularBuffer in_a(cb_a), in_b(cb_b), out(cb_out);
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    in_a.wait_front(1);
    in_b.wait_front(1);
    out.reserve_back(1);
    tile_regs_acquire();
    uint32_t out_dst = 0;
    if constexpr (OP == 0 || OP == 1) {
        matmul_init(cb_a, cb_b, OP);
        matmul_tiles(cb_a, cb_b, 0, 0, 0);
    } else if constexpr (OP == 2) {
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        gtz_tile_init();
        gtz_tile(0);
    } else if constexpr (OP == 3) {
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, SCALAR_1);
        mul_unary_tile(1, SCALAR_1);
        add_unary_tile(1, SCALAR_2);
        sqrt_tile_init();
        sqrt_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else if constexpr (OP == 4) {
        copy_init(cb_a);
        copy_tile(cb_a, 0, 6);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 7);
        add_binary_tile_init();
        add_binary_tile(6, 7, 6);
        out_dst = 6;
    } else if constexpr (OP == 6 || OP == 7) {
        out_dst = OP == 6 ? 5 : 1;
        copy_init(cb_a);
        copy_tile(cb_a, 0, out_dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(out_dst, SCALAR_1);
    } else {
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 0);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        relu_tile_init();
        relu_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(out_dst, cb_out, 0);
    tile_regs_release();
    out.push_back(1);
    in_a.pop_front(1);
    in_b.pop_front(1);
}
