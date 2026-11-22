// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Dataflow helpers of the packed verify passes (attn_common.hpp, group_rows > 1): the per-row rope tiles a pass's
// q (row r * heads + h from verify row g * group_rows + r) and the tail core's k rows (row r from verify row r)
// take, built from the token state's per-position rope tiles (rows replicated).
#pragma once

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

namespace resident_attn {

// cb_rope's pages kRopeQ .. kRopePages (the first 3 * batch pages, the token state's, already in place at base)
FORCE_INLINE void build_packed_rope(uint32_t base) {
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + kRopeQ * kTile);
    for (uint32_t i = 0; i < (kRopePages - kRopeQ) * kTile / 4; i++) {
        z[i] = 0;
    }
    // row 0 of position tile `src` (both face rows) into row `row` of tile `dst`
    auto copy_row = [&](uint32_t src, uint32_t dst, uint32_t row) {
        const uint32_t from = base + src * kTile, to = base + dst * kTile + row_offset(row);
        noc_async_read(get_noc_addr(from), to, kFaceRow);
        noc_async_read(get_noc_addr(from + kFace), to + kFace, kFaceRow);
    };
    // (pass ug = its user's pass g = ug % kPPU takes rows (ug / kPPU) kVR + g group_rows + r; user v's tail rope at
    // kRopeK + 3 v, row r from its row v kVR + r — a single-user verify: one user of kVR = batch rows)
    for (uint32_t k = 0; k < 3; k++) {
        for (uint32_t ug = 0; ug < kUsers; ug++) {
            const uint32_t r0 = (ug / kPPU) * kVR + (ug % kPPU) * group_rows;
            for (uint32_t r = 0; r < group_rows; r++) {
                for (uint32_t h = 0; h < heads; h++) {
                    copy_row(3 * (r0 + r) + k, kRopeQ + 3 * ug + k, r * heads + h);
                }
            }
        }
        for (uint32_t v = 0; v < kVUsers; v++) {
            for (uint32_t r = 0; r < kVR; r++) {
                copy_row(3 * (v * kVR + r) + k, kRopeK + 3 * v + k, r);
            }
        }
    }
    noc_async_read_barrier();
}

// row groups (kTailBatched): pages kRopeRows .. + 3 — row u of the cos / sin / -sin tiles from user u's page (rows
// replicated); the users' pages already in place at base
FORCE_INLINE void build_rows_rope(uint32_t base) {
    for (uint32_t k = 0; k < 3; k++) {
        for (uint32_t u = 0; u < kUsers; u++) {
            const uint32_t from = base + (3 * u + k) * kTile, to = base + (kRopeRows + k) * kTile + row_offset(u);
            noc_async_read(get_noc_addr(from), to, kFaceRow);
            noc_async_read(get_noc_addr(from + kFace), to + kFace, kFaceRow);
        }
    }
    noc_async_read_barrier();
}

}  // namespace resident_attn
