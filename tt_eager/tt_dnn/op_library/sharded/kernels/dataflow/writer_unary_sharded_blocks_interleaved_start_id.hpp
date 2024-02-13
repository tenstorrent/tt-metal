// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dataflow_api.h"

template <bool dst_is_dram>
inline void writer_unary_sharded_blocks_interleaved_start_id(
                                                        const uint32_t dst_addr,
                                                        const uint32_t block_height_tiles,
                                                        const uint32_t block_width_tiles,
                                                        const uint32_t unpadded_block_height_tiles,
                                                        const uint32_t unpadded_block_width_tiles,
                                                        const uint32_t output_width_tiles,
                                                        const uint32_t block_num_tiles,
                                                        const uint32_t start_id,
                                                        const uint32_t cb_id_out
                                                        ) {

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    const uint32_t padded_width_diff = (block_width_tiles - unpadded_block_width_tiles) * tile_bytes;

    uint32_t row_start_tile_id = start_id;
    cb_wait_front(cb_id_out, block_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    for (uint32_t h = 0; h < unpadded_block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < unpadded_block_width_tiles; w++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
            noc_async_write_barrier();
        }
        l1_read_addr += padded_width_diff;
        row_start_tile_id += output_width_tiles;
    }
    cb_pop_front(cb_id_out, block_num_tiles);
}
