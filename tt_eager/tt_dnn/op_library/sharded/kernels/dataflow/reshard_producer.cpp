// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.hpp"

void kernel_main() {
    const uint32_t num_units_input = get_arg_val<uint32_t>(0);
    const uint32_t dram_buffer_addr = get_arg_val<uint32_t>(1);

    //parameters to write to DRAM (matching input shards)
    const uint32_t block_height_tiles_to_dram = get_arg_val<uint32_t>(2);
    const uint32_t block_width_tiles_to_dram = get_arg_val<uint32_t>(3);
    const uint32_t unpadded_block_height_tiles_to_dram = get_arg_val<uint32_t>(4);
    const uint32_t unpadded_block_width_tiles_to_dram = get_arg_val<uint32_t>(5);
    const uint32_t output_width_tiles_to_dram = get_arg_val<uint32_t>(6); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles_to_dram = get_arg_val<uint32_t>(7); // block_height_tiles * block_width_tiles
    const uint32_t start_id_to_dram = get_arg_val<uint32_t>(8);


    //parameters to read from DRAM (matching output shards)
    const uint32_t block_height_tiles_from_dram = get_arg_val<uint32_t>(9);
    const uint32_t block_width_tiles_from_dram = get_arg_val<uint32_t>(10);
    const uint32_t input_width_offset_tiles_from_dram = get_arg_val<uint32_t>(11); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles_from_dram = get_arg_val<uint32_t>(12); // block_height_tiles * block_width_tiles
    const uint32_t start_id_from_dram = get_arg_val<uint32_t>(13);



    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);

    //signals that input shard is ready
    reader_unary_sharded(num_units_input, cb_id_in0);

    constexpr bool is_dram = true;

    // write input to temporary dram buffer,
    // can't use L1 buffer as new shards might come from other cores' L1s
    writer_unary_sharded_blocks_interleaved_start_id<is_dram>(dram_buffer_addr,
                                                    block_height_tiles_to_dram,
                                                    block_width_tiles_to_dram,
                                                    unpadded_block_height_tiles_to_dram,
                                                    unpadded_block_width_tiles_to_dram,
                                                    output_width_tiles_to_dram,
                                                    block_num_tiles_to_dram,
                                                    start_id_to_dram,
                                                    cb_id_in0
                                                    );

    //Once it's in DRAM write to CBs to be written
    //Should be safe since above API has an noc_async_write_barrier
    reader_unary_sharded_blocks_interleaved_start_id<is_dram>(
        dram_buffer_addr,
        block_height_tiles_from_dram,
        block_width_tiles_from_dram,
        input_width_offset_tiles_from_dram, // input width in tiles - block width in tiles
        block_num_tiles_from_dram, // block_height_tiles * block_width_tiles
        start_id_from_dram,
        cb_id_out0
    );


}
