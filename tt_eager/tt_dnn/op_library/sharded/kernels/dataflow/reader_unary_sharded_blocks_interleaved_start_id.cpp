// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.hpp"

void kernel_main() {
    const uint32_t src_addr  = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t input_width_offset_tiles = get_arg_val<uint32_t>(3); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(4); // block_height_tiles * block_width_tiles
    const uint32_t start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;


    reader_unary_sharded_blocks_interleaved_start_id <src_is_dram> (
        src_addr,
        block_height_tiles,
        block_width_tiles,
        input_width_offset_tiles, // input width in tiles - block width in tiles
        block_num_tiles, // block_height_tiles * block_width_tiles
        start_id,
        cb_id_in0
    );
}
