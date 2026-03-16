// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU elemwise backward kernel.

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

constexpr uint32_t cb_linear1 = tt::CBIndex::c_0;
constexpr uint32_t cb_gate = tt::CBIndex::c_1;
constexpr uint32_t cb_dL_dprod = tt::CBIndex::c_2;
constexpr uint32_t cb_dL_dlinear1 = tt::CBIndex::c_3;
constexpr uint32_t cb_dL_dgate = tt::CBIndex::c_4;
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_5;
constexpr uint32_t cb_scratch = tt::CBIndex::c_6;
constexpr uint32_t cb_silu_grad = tt::CBIndex::c_7;

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

inline void compute_dL_dgate() {
    cb_wait_front(cb_sigmoid, block_size);

    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_linear1, cb_sigmoid);
        mul_tiles(cb_linear1, cb_sigmoid, i, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

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

inline void compute_silu_grad() {
    const uint32_t one = 0x3F800000;

    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile_init(cb_sigmoid);
        copy_tile(cb_sigmoid, i, i);
        binop_with_scalar_tile_init();
        rsub_unary_tile(i, one);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

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

inline void compute_dL_dlinear1() {
    cb_wait_front(cb_silu_grad, block_size);

    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles_init(cb_gate, cb_dL_dprod);
        mul_tiles(cb_gate, cb_dL_dprod, i, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

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

            compute_sigmoid();
            compute_dL_dgate();
            compute_silu_grad();
            compute_dL_dlinear1();

            cb_pop_front(cb_sigmoid, block_size);
            cb_pop_front(cb_linear1, block_size);
            cb_pop_front(cb_gate, block_size);
            cb_pop_front(cb_dL_dprod, block_size);
        }
    }
}
