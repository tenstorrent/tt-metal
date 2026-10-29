// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernels only (NoC reads): the lanes' positions from the position inputs.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tile_rows.h"

namespace tile_rows {
// P of every lane from kv_block_start (uint32 ROW_MAJOR row, element u = P_u & ~31) and kv_row_hit (bf16 TILE, tile u
// = the one-hot column of P_u % 32).  scratch: >= 128 + rows * 2048 bytes; the result lands in positions[0 .. rows).
// Reads are issued and waited here.
template <typename BlockStartAccessor, typename HitAccessor>
inline void read_positions(
    CoreLocalMem<volatile uint32_t> positions,
    const BlockStartAccessor& block_start,
    const HitAccessor& hit,
    uint32_t rows,
    uint32_t scratch) {
    const Noc noc;
    const CoreLocalMem<uint32_t> scratch_l1(scratch);
    noc.async_read(block_start, scratch_l1, block_start.get_aligned_page_size(), {.page_id = 0}, {});
    for (uint32_t u = 0; u < rows; ++u) {
        noc.async_read(
            hit, scratch_l1, hit.get_aligned_page_size(), {.page_id = u}, {.offset_bytes = 128 + u * tile_bytes});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();
    const CoreLocalMem<const volatile uint32_t> starts(scratch);
    for (uint32_t u = 0; u < rows; ++u) {
        uint32_t row = 0;
        for (uint32_t r = 0; r < rows_per_tile; ++r) {
            const CoreLocalMem<const volatile uint16_t> hit_row(scratch + 128 + u * tile_bytes + chunk_offset(r, 0));
            if (hit_row[0] == 0x3F80) {
                row = r;
            }
        }
        positions[u] = starts[u] + row;
    }
}
}  // namespace tile_rows
