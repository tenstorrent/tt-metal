// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed-SwiGLU forward: h = silu(gate) * up over the two column halves of one tensor.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

constexpr uint32_t cb_gate = tt::CBIndex::c_0;
constexpr uint32_t cb_up = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_silu = tt::CBIndex::c_3;
constexpr uint32_t cb_scratch = tt::CBIndex::c_4;  // TRANSPOSED_OUTPUT: the product before its transpose

// h = silu(gate) * up for one block.
inline void swiglu_block() {
    tile_regs_acquire();
    copy_init(cb_gate);
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile(cb_gate, i, i);
    }
    silu_tile_init();
    for (uint32_t i = 0; i < block_size; ++i) {
        silu_tile(i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_silu, block_size);

    cb_wait_front(cb_silu, block_size);
    tile_regs_acquire();
    mul_init(cb_silu, cb_up);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_silu, cb_up, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_silu, block_size);
#ifdef TRANSPOSED_OUTPUT
    // h^T (iteration 38): the product once more through DEST, transposed (exact), for the writer's column scatter
    pack_and_push_block(cb_scratch, block_size);
    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    transpose_init(cb_scratch);
    for (uint32_t i = 0; i < block_size; ++i) {
        transpose_tile(cb_scratch, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    pack_and_push_block(cb_out, block_size);
    copy_init(cb_gate);  // the next block starts with copy_tile from cb_gate
#else
    pack_and_push_block(cb_out, block_size);
#endif
}

void kernel_main() {
    compute_kernel_hw_startup(cb_gate, cb_up, cb_out);
    copy_init(cb_gate);

    for (uint32_t block = 0; block < num_blocks_per_core; ++block) {
        cb_wait_front(cb_gate, block_size);
        cb_wait_front(cb_up, block_size);

        swiglu_block();

        cb_pop_front(cb_gate, block_size);
        cb_pop_front(cb_up, block_size);
    }
}
