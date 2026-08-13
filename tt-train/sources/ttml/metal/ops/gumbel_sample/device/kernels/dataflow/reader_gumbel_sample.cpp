// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams logits (and, when present, the additive padding mask) one block of vocab tiles at a time.
// Nothing is buffered for a whole row, so L1 use is independent of V.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t logits_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t mask_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_logits_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_1;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

#ifdef DO_LOGITS_MASK
    constexpr bool do_logits_mask = true;
#else
    constexpr bool do_logits_mask = false;
#endif

    constexpr auto logits_args = TensorAccessorArgs<2>();
    constexpr auto mask_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    const auto logits_address_generator = TensorAccessor(logits_args, logits_address);
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);

    const uint32_t logits_tile_bytes = get_tile_size(cb_logits_idx);

    for (uint32_t i = 0U; i < num_rows_to_process; ++i) {
        // Tile index of the first vocab tile of this 32-token tile row.
        const uint32_t row_start_idx = (start_row + i) * Wt;

        // block_size divides Wt, so every block is full.
        for (uint32_t j = 0U; j < Wt; j += block_size) {
            read_tiles_by_row(
                cb_logits_idx, logits_address_generator, row_start_idx + j, block_size, logits_tile_bytes, block_size);

            if constexpr (do_logits_mask) {
                // The mask is [1, 1, 1, V]: a single tile row that every token row reuses, so its
                // tile ids are just the column index and must NOT advance with the logits' row
                // offset. The compute kernel then broadcasts row 0 down the tile.
                read_tiles_by_row(cb_mask_idx, mask_address_generator, j, block_size, logits_tile_bytes, block_size);
            }
        }
    }
}
