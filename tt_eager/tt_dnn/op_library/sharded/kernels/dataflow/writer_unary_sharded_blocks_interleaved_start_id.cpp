// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.hpp"

void kernel_main() {
    const uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t unpadded_block_height_tiles = get_arg_val<uint32_t>(3);
    const uint32_t unpadded_block_width_tiles = get_arg_val<uint32_t>(4);
    const uint32_t output_width_tiles = get_arg_val<uint32_t>(5); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(6); // block_height_tiles * block_width_tiles
    const uint32_t start_id = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;


    writer_unary_sharded_blocks_interleaved_start_id <dst_is_dram> (dst_addr,
                                                    block_height_tiles,
                                                    block_width_tiles,
                                                    unpadded_block_height_tiles,
                                                    unpadded_block_width_tiles,
                                                    output_width_tiles,
                                                    block_num_tiles,
                                                    start_id,
                                                    cb_id_out
                                                    );

}
