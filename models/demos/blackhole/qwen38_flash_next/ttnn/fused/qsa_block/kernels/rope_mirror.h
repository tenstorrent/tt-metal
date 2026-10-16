// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// rotary_embedding_hf.cpp for 64-wide row tiles (Wt 2, half_Wt 1): rotated * (-1) [bcast scalar], * sin, x * cos,
// add; FPU ops, each packed to a bf16 CB.  Bitwise against the op only under the op's own 16-bit dest (the FPU
// rounds into a 16-bit dest by a rule no fp32-dest reproduction matched: fused_qsa_llk_pins, 2026-09-14), so the
// kernel must be compiled with fp32_dest_acc_en off.  The rotated input is the two tiles in swapped order.
// CBs: c_in x (bf16, 2), c_rot rotated x (bf16, 2), c_cos, c_sin (bf16, 2 each, popped), c_scalar (bf16, 1, never
// popped; element [0,0] = -1), c_rot_neg, c_xcos, c_rsin (bf16, 1 each), c_out (bf16, 2).  The caller runs
// compute_kernel_hw_startup(c_rot, c_scalar, c_rot_neg) once and waits on c_scalar.

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

template <uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id>
inline void rope_mul_tiles() {
    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);
    in0_cb.wait_front(1);
    in1_cb.wait_front(1);
    out_cb.reserve_back(1);
    tile_regs_acquire();
    mul_init(in0_cb_id, in1_cb_id);
    mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb_id);
    tile_regs_release();
    out_cb.push_back(1);
    in0_cb.pop_front(1);
    in1_cb.pop_front(1);
}

template <
    uint32_t c_in,
    uint32_t c_rot,
    uint32_t c_cos,
    uint32_t c_sin,
    uint32_t c_scalar,
    uint32_t c_rot_neg,
    uint32_t c_xcos,
    uint32_t c_rsin,
    uint32_t c_out>
inline void rope64_rows(uint32_t rows) {
    constexpr uint32_t Wt = 2, half_Wt = 1;
    CircularBuffer rot_cb(c_rot), rot_neg_cb(c_rot_neg), xcos_cb(c_xcos), rsin_cb(c_rsin), out_cb(c_out);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                reconfig_data_format(c_rot, c_scalar);
                pack_reconfig_data_format(c_rot_neg);
                rot_cb.wait_front(1);
                rot_neg_cb.reserve_back(1);
                tile_regs_acquire();
                mul_bcast_scalar_init(c_rot, c_scalar);
                mul_tiles_bcast_scalar(c_rot, c_scalar, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, c_rot_neg);
                tile_regs_release();
                rot_neg_cb.push_back(1);
                rot_cb.pop_front(1);
                reconfig_data_format_srcb(c_scalar, c_sin);
                pack_reconfig_data_format(c_rot_neg, c_rsin);
                rope_mul_tiles<c_rot_neg, c_sin, c_rsin>();
            } else {
                reconfig_data_format(c_rot, c_sin);
                pack_reconfig_data_format(c_out, c_rsin);
                rope_mul_tiles<c_rot, c_sin, c_rsin>();
            }
            rope_mul_tiles<c_in, c_cos, c_xcos>();

            xcos_cb.wait_front(1);
            rsin_cb.wait_front(1);
            out_cb.reserve_back(1);
            reconfig_data_format_srca(c_rot, c_xcos);
            pack_reconfig_data_format(c_xcos, c_out);
            tile_regs_acquire();
            add_init(c_xcos, c_rsin);
            add_tiles(c_xcos, c_rsin, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, c_out);
            tile_regs_release();
            out_cb.push_back(1);
            xcos_cb.pop_front(1);
            rsin_cb.pop_front(1);
        }
    }
}
