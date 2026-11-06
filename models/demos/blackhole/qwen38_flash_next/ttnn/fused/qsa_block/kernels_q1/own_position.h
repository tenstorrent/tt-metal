// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernels only (NoC reads): one lane's position from the chain's position inputs (QWEN38_QSA_OWNPOS).
// read_positions (../kernels/positions.h) reads every lane's hit tile and scans them all although a core uses only
// positions[lane]; this reads the block-start row (one page) and the lane's own hit tile only.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "../kernels/tile_rows.h"

namespace tile_rows {
// P of `lane`: kv_block_start (uint32 ROW_MAJOR row, element u = P_u & ~31) + the row of the hit tile `lane` whose
// column 0 is 1.0 (the last such row, 0 if none: as read_positions).  The value is stored in positions[lane] (no other
// entry is written) and returned.  scratch: >= 128 + 2048 bytes.  Reads are issued and waited here.
template <typename BlockStartAccessor, typename HitAccessor>
inline uint32_t read_own_position(
    volatile tt_l1_ptr uint32_t* positions,
    const BlockStartAccessor& block_start,
    const HitAccessor& hit,
    uint32_t lane,
    uint32_t scratch) {
    noc_async_read_page(0, block_start, scratch);
    noc_async_read_page(lane, hit, scratch + 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* starts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t row = 0;
    for (uint32_t r = 0; r < TILE_ROWS; ++r) {
        const volatile tt_l1_ptr uint16_t* e =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 128 + chunk_offset(r, 0));
        if (e[0] == 0x3F80) {
            row = r;
        }
    }
    const uint32_t p = starts[lane] + row;
    positions[lane] = p;
    return p;
}
}  // namespace tile_rows
