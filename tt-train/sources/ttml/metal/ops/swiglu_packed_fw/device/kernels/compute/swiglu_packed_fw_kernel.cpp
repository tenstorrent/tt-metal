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
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

constexpr uint32_t cb_gate = tt::CBIndex::c_0;
constexpr uint32_t cb_up = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_silu = tt::CBIndex::c_3;

// h = silu(gate) * up for one block.
inline void swiglu_block() {
    tile_regs_acquire();
    copy_tile_init(cb_gate);
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
    pack_and_push_block(cb_out, block_size);
}

void kernel_main() {
    init_sfpu(cb_gate, cb_out);
    // TODO(#52395): compute_kernel_hw_startup is a call-once API and should be the kernel's first Tensix-engine call,
    // but here it follows another engine op (init_sfpu / a prior startup); see the issue.
    compute_kernel_hw_startup(cb_gate, cb_up, cb_out);

    for (uint32_t block = 0; block < num_blocks_per_core; ++block) {
        cb_wait_front(cb_gate, block_size);
        cb_wait_front(cb_up, block_size);

        swiglu_block();

        cb_pop_front(cb_gate, block_size);
        cb_pop_front(cb_up, block_size);
    }
}
