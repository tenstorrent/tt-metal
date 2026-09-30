// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RoPE of one head's rope tail (RT tiles of one tile row) for the V4.1 head-layout kernels (heads_q_compute.cpp,
// heads_o_compute.cpp): out = x * cos + (x @ trans) * sin, the op sequence of ttnn's rotary_embedding_llama compute
// kernel with its bf16 intermediates (rotated, sin and cos products), so the result matches that op bit for bit.
// Inverse RoPE passes -sin. All operands bf16 tiles; cos / sin hold the unit's tile row (RT tiles, waited on by the
// caller), trans the 32x32 rotation (one tile, waited on by the caller).

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"

template <
    uint32_t RT,
    uint32_t cb_x,
    uint32_t cb_cos,
    uint32_t cb_sin,
    uint32_t cb_trans,
    uint32_t cb_rot,
    uint32_t cb_sini,
    uint32_t cb_cosi,
    uint32_t cb_out>
ALWI void rope_tail() {
    DataflowBuffer x(cb_x);
    DataflowBuffer rot(cb_rot);
    DataflowBuffer sini(cb_sini);
    DataflowBuffer cosi(cb_cosi);
    DataflowBuffer out(cb_out);

    x.wait_front(RT);
    // rotated = x @ trans
    rot.reserve_back(RT);
    matmul_init(cb_x, cb_trans);
    tile_regs_acquire();
    for (uint32_t j = 0; j < RT; ++j) {
        matmul_tiles(cb_x, cb_trans, j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < RT; ++j) {
        pack_tile(j, cb_rot, j);
    }
    tile_regs_release();
    rot.push_back(RT);

    // sin_interm = rotated * sin
    rot.wait_front(RT);
    sini.reserve_back(RT);
    mul_init(cb_rot, cb_sin, false);
    tile_regs_acquire();
    for (uint32_t j = 0; j < RT; ++j) {
        mul_tiles(cb_rot, cb_sin, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < RT; ++j) {
        pack_tile(j, cb_sini, j);
    }
    tile_regs_release();
    sini.push_back(RT);
    rot.pop_front(RT);

    // cos_interm = x * cos
    cosi.reserve_back(RT);
    mul_init(cb_x, cb_cos, false);
    tile_regs_acquire();
    for (uint32_t j = 0; j < RT; ++j) {
        mul_tiles(cb_x, cb_cos, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < RT; ++j) {
        pack_tile(j, cb_cosi, j);
    }
    tile_regs_release();
    cosi.push_back(RT);
    x.pop_front(RT);

    // out = cos_interm + sin_interm
    sini.wait_front(RT);
    cosi.wait_front(RT);
    out.reserve_back(RT);
    add_init(cb_cosi, cb_sini);
    tile_regs_acquire();
    for (uint32_t j = 0; j < RT; ++j) {
        add_tiles(cb_cosi, cb_sini, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < RT; ++j) {
        pack_tile(j, cb_out, j);
    }
    tile_regs_release();
    out.push_back(RT);
    sini.pop_front(RT);
    cosi.pop_front(RT);
}
