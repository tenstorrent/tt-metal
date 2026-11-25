// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared compute for the SwiGLU-gating backward, used by both `swiglu_elemwise_bw` (two separate
// [.,I] tensors) and `swiglu_packed_bw` (one packed [.,2I] tensor).
//
// Given the gate branch (the one that is silu'd), the up branch, and the upstream grad dL/dh:
//   dL/d(up)   = dL/dh * silu(gate)
//   dL/d(gate) = (dL/dh * up) * silu'(gate)
//
// Both gradients stay in DEST from the first SFPU step to the pack: silu(gate) by silu_tile, then the FPU multiplies
// dL/dh in from DEST; silu'(gate) by silu_bw_tile, the same SFPU kernel as ttnn::silu_bw, applied to the incoming
// gradient dL/dh * up. Each output is rounded once more than its SFPU result, instead of once per step of
// sigmoid * (1 + gate * (1 - sigmoid)) through circular buffers. silu_bw_tile evaluates a BF16 DEST, so the
// programs run with fp32_dest_acc_en = false, and it reads x and the gradient from two DEST tiles, so a block of
// up to 4 tiles uses the 8 DEST tiles of a half-sync BF16 DEST.
//
// KERNEL-SIDE ONLY: include this from a compute kernel .cpp; it pulls in the LLK compute API.

#pragma once

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reg_api.h"
#include "api/compute/silu_bw.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

// Compute one block of the SwiGLU-gating backward. The caller must have already
// waited `block_size` tiles on cb_gate / cb_up / cb_dh; this function leaves cb_grad_gate / cb_grad_up pushed.
// Input CBs are popped by the caller. cb_sigmoid / cb_scratch / cb_silu_grad are no longer used.
template <
    uint32_t cb_gate,       // gate branch (silu'd)
    uint32_t cb_up,         // up branch
    uint32_t cb_dh,         // upstream grad dL/dh
    uint32_t cb_grad_gate,  // out: grad wrt gate branch
    uint32_t cb_grad_up,    // out: grad wrt up branch
    uint32_t cb_sigmoid,
    uint32_t cb_scratch,
    uint32_t cb_silu_grad,
    uint32_t block_size>
inline void swiglu_gate_bw_block() {
    static_assert(2 * block_size <= 8, "silu_bw_tile needs two BF16 DEST tiles per tile of the block");

    // dL/d(up) = silu(gate) * dL/dh: silu in DEST, then dL/dh multiplied in from DEST.
    tile_regs_acquire();
    copy_init(cb_gate);
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile(cb_gate, i, i);
    }
    silu_tile_init();
    for (uint32_t i = 0; i < block_size; ++i) {
        silu_tile(i);
    }
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_dh);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_dh, i, i);
    }
    tile_regs_commit();
    pack_and_push_block(cb_grad_up, block_size);

    // dL/d(gate) = (dL/dh * up) * silu'(gate): silu_bw_tile(2i) reads gate from DEST 2i and the incoming gradient
    // from DEST 2i + 1, and writes the product into DEST 2i.
    tile_regs_acquire();
    copy_init(cb_gate);
    for (uint32_t i = 0; i < block_size; ++i) {
        copy_tile(cb_gate, i, 2 * i);
    }
    mul_init(cb_up, cb_dh);
    for (uint32_t i = 0; i < block_size; ++i) {
        mul_tiles(cb_up, cb_dh, i, i, 2 * i + 1);
    }
    silu_bw_tile_init();
    for (uint32_t i = 0; i < block_size; ++i) {
        silu_bw_tile(2 * i);
    }
    tile_regs_commit();

    cb_reserve_back(cb_grad_gate, block_size);
    tile_regs_wait();
    pack_reconfig_data_format(cb_grad_gate);
    for (uint32_t i = 0; i < block_size; ++i) {
        pack_tile(2 * i, cb_grad_gate);
    }
    tile_regs_release();
    cb_push_back(cb_grad_gate, block_size);
}
