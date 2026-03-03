// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU gradient kernel.
// Inputs:  linear1 [B,N,S,H], gate [B,N,S,H], dL_dprod [B,N,S,H]
// Outputs: dL_dlinear1 [B,N,S,H], dL_dgate [B,N,S,H]
//
// Per-element:
//   sig         = sigmoid(linear1)
//   swished     = linear1 * sig
//   dL_dgate    = swished * dL_dprod              (output 2)
//   dL_dswished = gate * dL_dprod
//   silu_grad   = sig * (linear1 * (1-sig) + 1)
//   dL_dlinear1 = dL_dswished * silu_grad         (output 1)

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

// Input CBs
constexpr uint32_t cb_linear1 = tt::CBIndex::c_0;
constexpr uint32_t cb_gate = tt::CBIndex::c_1;
constexpr uint32_t cb_dL_dprod = tt::CBIndex::c_2;
// Output CBs
constexpr uint32_t cb_dL_dlinear1 = tt::CBIndex::c_3;
constexpr uint32_t cb_dL_dgate = tt::CBIndex::c_4;
// Intermediate CBs
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_5;
constexpr uint32_t cb_scratch = tt::CBIndex::c_6;
constexpr uint32_t cb_silu_grad = tt::CBIndex::c_7;

// Step 1: sigmoid(linear1) → cb_sigmoid
inline void compute_sigmoid() {
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile_init(cb_linear1);
        copy_tile(cb_linear1, i, i);
        sigmoid_tile_init();
        sigmoid_tile(i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_sigmoid, block_size);
}

// Step 2: dL_dgate = silu(linear1) * dL_dprod = (linear1 * sigmoid) * dL_dprod
// Computes swished → cb_scratch, then swished * dL_dprod → cb_dL_dgate
inline void compute_dL_dgate() {
    cb_wait_front(cb_sigmoid, block_size);

    // swished = linear1 * sigmoid → cb_scratch
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_linear1, cb_sigmoid);
        mul_tiles(cb_linear1, cb_sigmoid, i, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

    // dL_dgate = swished * dL_dprod → cb_dL_dgate (FINAL output)
    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_scratch, cb_dL_dprod);
        mul_tiles(cb_scratch, cb_dL_dprod, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    pack_and_push_block(cb_dL_dgate, block_size);
}

// Step 3: silu_grad = sigmoid * (linear1 * (1 - sigmoid) + 1) → cb_silu_grad
inline void compute_silu_grad() {
    const uint32_t one = 0x3F800000;

    // 1 - sigmoid → cb_scratch
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile_init(cb_sigmoid);
        copy_tile(cb_sigmoid, i, i);
        binop_with_scalar_tile_init();
        rsub_unary_tile(i, one);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

    // linear1 * (1 - sigmoid) + 1 → cb_silu_grad
    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_linear1, cb_scratch);
        mul_tiles(cb_linear1, cb_scratch, i, i, i);
        binop_with_scalar_tile_init();
        add_unary_tile(i, one);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    pack_and_push_block(cb_silu_grad, block_size);

    // sigmoid * (linear1*(1-sigmoid)+1) → cb_silu_grad (overwrite)
    cb_wait_front(cb_silu_grad, block_size);
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_sigmoid, cb_silu_grad);
        mul_tiles(cb_sigmoid, cb_silu_grad, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_silu_grad, block_size);
    pack_and_push_block(cb_silu_grad, block_size);
}

// Step 4: dL_dlinear1 = (gate * dL_dprod) * silu_grad
inline void compute_dL_dlinear1() {
    cb_wait_front(cb_silu_grad, block_size);

    // dL_dswished = gate * dL_dprod → cb_scratch
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_gate, cb_dL_dprod);
        mul_tiles(cb_gate, cb_dL_dprod, i, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

    // dL_dlinear1 = dL_dswished * silu_grad → cb_dL_dlinear1 (FINAL output)
    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_scratch, cb_silu_grad);
        mul_tiles(cb_scratch, cb_silu_grad, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    cb_pop_front(cb_silu_grad, block_size);
    pack_and_push_block(cb_dL_dlinear1, block_size);
}

void kernel_main() {
    init_sfpu(cb_linear1, cb_dL_dlinear1);
    binary_op_init_common(cb_linear1, cb_gate, cb_dL_dlinear1);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_linear1, block_size);
            cb_wait_front(cb_gate, block_size);
            cb_wait_front(cb_dL_dprod, block_size);

            compute_sigmoid();      // → cb_sigmoid
            compute_dL_dgate();     // → cb_dL_dgate (output, writer can consume)
            compute_silu_grad();    // → cb_silu_grad
            compute_dL_dlinear1();  // → cb_dL_dlinear1 (output, writer can consume)

            cb_pop_front(cb_sigmoid, block_size);
            cb_pop_front(cb_linear1, block_size);
            cb_pop_front(cb_gate, block_size);
            cb_pop_front(cb_dL_dprod, block_size);
        }
    }
}
