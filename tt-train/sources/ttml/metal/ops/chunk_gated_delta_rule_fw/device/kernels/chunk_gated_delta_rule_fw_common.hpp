// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Circular-buffer plan of the two forward programs, included by the host factories and the kernels.
// Prep and scan are separate dispatches, so they number their circular buffers independently.

#pragma once

#include <cstdint>

namespace chunk_gated_delta_rule_fw {

namespace prep {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7, cb_S = 8;
constexpr uint32_t cb_decay = 9, cb_decay_exp = 10, cb_decayfac = 11;
constexpr uint32_t cb_lmask = 12, cb_Tinv = 13, cb_vbeta = 14, cb_kbeta = 15;
constexpr uint32_t cb_out = 16, cb_u = 17, cb_w = 18, cb_qdecay = 19;
constexpr uint32_t cb_intra = 20, cb_s2 = 21, cb_vnew = 22, cb_ointer = 23;
constexpr uint32_t cb_kdec_t = 24, cb_supd = 25, cb_stmp = 26, cb_final = 27;
constexpr uint32_t cb_scr1 = 28, cb_scr2 = 29, cb_scr3 = 30, cb_s3 = 31;
// Role aliases: prep never uses these slots for their scan-side meaning.
constexpr uint32_t cb_gcum = cb_out;  // chunk-local cumsum(g), drained as g_cumsum
constexpr uint32_t cb_mask = cb_u;    // the three 16x16 quadrant masks of the WY inverse
constexpr uint32_t cb_kd = cb_w;      // k * beta * exp(cumsum g), drained for the scan
constexpr uint32_t cb_dl = cb_vnew;   // exp(sum of g over the chunk), drained for the scan
constexpr uint32_t kNumMasks = 3;     // top-left, bottom-right, bottom-left 16x16 quadrant
}  // namespace prep

namespace scan {
constexpr uint32_t cb_hout = 0, cb_vnout = 1, cb_h0 = 2;
constexpr uint32_t cb_S = 8, cb_dl = 11, cb_Tinv = 13, cb_out = 16;
constexpr uint32_t cb_vbeta = 17, cb_kd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_s2 = 21, cb_vnew = 22, cb_ointer = 23, cb_kdec_t = 24;
constexpr uint32_t cb_supd = 25, cb_stmp = 26, cb_final = 27, cb_scr1 = 28, cb_s3 = 31;
}  // namespace scan

// Element offset of (r, c) inside a 32x32 tile stored as four 16x16 faces.
constexpr uint32_t tile_elem_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256U + ((r & 15U) << 4) + (c & 15U);
}

}  // namespace chunk_gated_delta_rule_fw
