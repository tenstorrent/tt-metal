// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared compute for the SwiGLU-gating backward, used by both `swiglu_elemwise_bw` (two separate
// [.,I] tensors) and `swiglu_packed_bw` (one packed [.,2I] tensor).
//
// Given the gate branch (the one that is silu'd), the plain branch, and the upstream grad dL/dh:
//   dL/d(plain) = dL/dh * silu(gate),                       silu(gate) = gate * sigmoid(gate)
//   dL/d(gate)  = dL/dh * plain * silu'(gate),
//                 silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
//
// KERNEL-SIDE ONLY: include this from a compute kernel .cpp; it pulls in the LLK compute API.

#pragma once

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

// Compute one block of the SwiGLU-gating backward. The caller must have already
// waited `block_size` tiles on cb_gate / cb_plain / cb_dh; this function fully manages the internal
// cb_sigmoid / cb_scratch / cb_silu_grad and leaves cb_grad_gate / cb_grad_plain pushed. Input CBs
// are popped by the caller.
template <
    uint32_t cb_gate,        // gate branch (silu'd)
    uint32_t cb_plain,       // plain branch
    uint32_t cb_dh,          // upstream grad dL/dh
    uint32_t cb_grad_gate,   // out: grad wrt gate branch
    uint32_t cb_grad_plain,  // out: grad wrt plain branch
    uint32_t cb_sigmoid,
    uint32_t cb_scratch,
    uint32_t cb_silu_grad,
    uint32_t block_size>
inline void swiglu_gate_bw_block() {
    constexpr uint32_t one = 0x3F800000;  // 1.0f bits

    // sigmoid(gate) -> cb_sigmoid, stored for reuse in both gradients.
    tile_regs_acquire();
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile_init(cb_gate);
        copy_tile(cb_gate, i, i);
        sigmoid_tile_init();
        sigmoid_tile(i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_sigmoid, block_size);

    // dL/d(plain) = dL/dh * silu(gate), silu(gate) = gate * sigmoid(gate).
    cb_wait_front(cb_sigmoid, block_size);
    tile_regs_acquire();
    mul_init(cb_gate, cb_sigmoid);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_gate, cb_sigmoid, i, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    mul_init(cb_scratch, cb_dh);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_scratch, cb_dh, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    pack_and_push_block(cb_grad_plain, block_size);

    // silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))).
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
    mul_init(cb_gate, cb_scratch);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_gate, cb_scratch, i, i, i);
        binop_with_scalar_tile_init();
        add_unary_tile(i, one);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    pack_and_push_block(cb_silu_grad, block_size);

    cb_wait_front(cb_silu_grad, block_size);
    tile_regs_acquire();
    mul_init(cb_sigmoid, cb_silu_grad);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_sigmoid, cb_silu_grad, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_silu_grad, block_size);
    pack_and_push_block(cb_silu_grad, block_size);

    // dL/d(gate) = dL/dh * plain * silu'(gate).
    cb_wait_front(cb_silu_grad, block_size);
    tile_regs_acquire();
    mul_init(cb_plain, cb_dh);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_plain, cb_dh, i, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_scratch, block_size);

    cb_wait_front(cb_scratch, block_size);
    tile_regs_acquire();
    mul_init(cb_scratch, cb_silu_grad);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_scratch, cb_silu_grad, i, i, i);
    }
    tile_regs_commit();
    cb_pop_front(cb_scratch, block_size);
    cb_pop_front(cb_silu_grad, block_size);
    pack_and_push_block(cb_grad_gate, block_size);

    cb_pop_front(cb_sigmoid, block_size);
}
