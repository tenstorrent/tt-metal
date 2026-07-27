// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed-SwiGLU forward: h = silu(gate) * up over the two column halves of one tensor.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

constexpr uint32_t cb_gate = tt::CBIndex::c_0;
constexpr uint32_t cb_up = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_3;
constexpr uint32_t cb_scratch = tt::CBIndex::c_4;

// sigmoid(gate) -> cb_sigmoid
inline void compute_sigmoid() {
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile_init(cb_gate);
        copy_tile(cb_gate, i, i);
        sigmoid_tile_init();
        sigmoid_tile(i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_sigmoid, block_size);
}

// silu(gate) = gate * sigmoid(gate) -> cb_scratch
inline void compute_silu() {
    cb_wait_front(cb_sigmoid, block_size);
    tile_regs_acquire();
    mul_tiles_init(cb_gate, cb_sigmoid);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_gate, cb_sigmoid, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_sigmoid, block_size);
    pack_and_push_block(cb_scratch, block_size);
}

// h = silu(gate) * up -> cb_out
inline void compute_out() {
    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    mul_tiles_init(cb_scratch, cb_up);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_scratch, cb_up, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    pack_and_push_block(cb_out, block_size);
}

void kernel_main() {
    init_sfpu(cb_gate, cb_out);
    binary_op_init_common(cb_gate, cb_up, cb_out);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_gate, block_size);
            cb_wait_front(cb_up, block_size);

            compute_sigmoid();
            compute_silu();
            compute_out();

            cb_pop_front(cb_gate, block_size);
            cb_pop_front(cb_up, block_size);
        }
    }
}
