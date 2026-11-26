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
// One SFPU pass (swiglu_gate_bw_tile, swiglu_gate_bw_sfpu.h) computes both from gate, dL/dh and up in DEST, with one
// sigmoid, in fp32, and rounds each to BF16 once. It reads three DEST tiles per tile of the block and needs a BF16
// DEST, so the programs run with fp32_dest_acc_en = false and a block goes through DEST two tiles at a time.
//
// KERNEL-SIDE ONLY: include this from a compute kernel .cpp; it pulls in the LLK compute API.

#pragma once

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/swiglu_gate_bw_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// Compute one block of the SwiGLU-gating backward. The caller must have already
// waited `block_size` tiles on cb_gate / cb_up / cb_dh; this function leaves cb_grad_gate / cb_grad_up pushed.
// Input CBs are popped by the caller.
template <
    uint32_t cb_gate,       // gate branch (silu'd)
    uint32_t cb_up,         // up branch
    uint32_t cb_dh,         // upstream grad dL/dh
    uint32_t cb_grad_gate,  // out: grad wrt gate branch
    uint32_t cb_grad_up,    // out: grad wrt up branch
    uint32_t block_size>
inline void swiglu_gate_bw_block() {
    // gate, dL/dh and up of one tile take DEST tiles 3k, 3k + 1 and 3k + 2.
    constexpr uint32_t tiles_per_pass = compute_kernel_lib::DEST_AUTO_LIMIT / 3U;
    static_assert(tiles_per_pass >= 1U, "swiglu_gate_bw_tile needs three DEST tiles");

    for (uint32_t first = 0; first < block_size; first += tiles_per_pass) {
        const uint32_t n = block_size - first < tiles_per_pass ? block_size - first : tiles_per_pass;

        tile_regs_acquire();
        copy_init(cb_gate);
        for (uint32_t k = 0; k < n; ++k) {
            copy_tile(cb_gate, first + k, 3U * k);
        }
        copy_init(cb_dh);
        for (uint32_t k = 0; k < n; ++k) {
            copy_tile(cb_dh, first + k, 3U * k + 1U);
        }
        copy_init(cb_up);
        for (uint32_t k = 0; k < n; ++k) {
            copy_tile(cb_up, first + k, 3U * k + 2U);
        }
        swiglu_gate_bw_tile_init();
        for (uint32_t k = 0; k < n; ++k) {
            swiglu_gate_bw_tile(3U * k);
        }
        tile_regs_commit();

        cb_reserve_back(cb_grad_gate, n);
        cb_reserve_back(cb_grad_up, n);
        tile_regs_wait();
        pack_reconfig_data_format(cb_grad_gate);
        for (uint32_t k = 0; k < n; ++k) {
            pack_tile(3U * k, cb_grad_gate);
        }
        pack_reconfig_data_format(cb_grad_up);
        for (uint32_t k = 0; k < n; ++k) {
            pack_tile(3U * k + 1U, cb_grad_up);
        }
        tile_regs_release();
        cb_push_back(cb_grad_gate, n);
        cb_push_back(cb_grad_up, n);
    }
}
