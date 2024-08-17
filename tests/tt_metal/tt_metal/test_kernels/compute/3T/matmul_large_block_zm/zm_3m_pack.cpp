// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"
namespace NAMESPACE
{

inline void tilize_activation(
    uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks, uint32_t in0_block_num_tiles, uint32_t matmul_act_cb_id) {
    llk_wait_for_free_tiles<false,false,false>(matmul_act_cb_id, in0_block_num_tiles);
    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t i = 0U; i < in0_subblock_h; i++) {
            for (uint32_t j = 0U; j < in0_block_w; j++) {
                llk_packer_wait_for_math_done();
                llk_pack<false, false >(0, matmul_act_cb_id);
                llk_pack_dest_section_done();
                llk_push_tiles<false,false>(matmul_act_cb_id, 1);
            }
        }
    }
}

inline void pack_row(uint32_t num_tiles_to_pack, uint32_t cb_id) {
    /*
        Used either for packing reblocked tiles for untilized tiles
    */
    llk_wait_for_free_tiles<false,false,false>(cb_id, num_tiles_to_pack);
    for (uint32_t i = 0; i < num_tiles_to_pack; i++) {
        llk_packer_wait_for_math_done();
        llk_pack<false, false >(0, cb_id);
        llk_pack_dest_section_done();
    }
    llk_push_tiles<false,false>(cb_id, num_tiles_to_pack);
}

inline void reblock_and_untilize_output(uint32_t out_subblock_h, uint32_t out_block_w, uint32_t reblock_cb_id, uint32_t untilize_cb_id) {
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        // Can only push row because the CB can only fit
        // one row
        pack_row(out_block_w, reblock_cb_id);
        pack_row(out_block_w, untilize_cb_id);
    }
}

inline void pack_block_and_untilize(
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t out_subblock_num_tiles, uint32_t out_subblock_h, uint32_t out_block_w,
    uint32_t interm_cb_id, uint32_t reblock_cb_id) {

    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
            llk_packer_wait_for_math_done();

            llk_wait_for_free_tiles<false,false,false>(interm_cb_id, out_subblock_num_tiles);
            for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
                llk_pack<false, false >(i, interm_cb_id);
            }
            llk_push_tiles<false,false>(interm_cb_id, out_subblock_num_tiles);
            llk_pack_dest_section_done();
        }
        reblock_and_untilize_output(out_subblock_h, out_block_w, reblock_cb_id, 16);
    }
}

inline void pack_block(uint32_t in0_num_subblocks, uint32_t in1_num_subblocks, uint32_t out_subblock_num_tiles, uint32_t cb_id) {

    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
            llk_packer_wait_for_math_done();

            llk_wait_for_free_tiles<false,false,false>(cb_id, out_subblock_num_tiles);
            for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
                llk_pack<false, false >(i, cb_id);
            }
            llk_push_tiles<false,false>(cb_id, out_subblock_num_tiles);
            llk_pack_dest_section_done();
        }
    }
}


void pack_main()
{
uint32_t in0_block_w = get_compile_time_arg_val(0);
llk_pack_init();
llk_pack_dest_init<DstTileFaceLayout::RowMajor, false>();
llk_init_packer_dest_offset_registers<DstTileFaceLayout::RowMajor,false>();
llk_pack_hw_configure_disaggregated<false>(16);
// inner block size in tiles
uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
// outer row block size (in inner row blocks)
uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
// out_subblock_h*in0_block_w*in0_num_subblocks;
uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
uint32_t in0_subblock_h = get_compile_time_arg_val(4);
uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
uint32_t in1_per_core_w = get_compile_time_arg_val(7);
constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
uint32_t out_subblock_h = get_compile_time_arg_val(9);
uint32_t out_subblock_w = get_compile_time_arg_val(10);
uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);

uint32_t out_block_w = in1_per_core_w;

// If true, this assumes data coming in RM
constexpr bool tilize_in = get_compile_time_arg_val(12);

// If true, this assumes consumer wants data RM
constexpr bool untilize_out = get_compile_time_arg_val(13);

constexpr bool spill = num_blocks > 1U;
bool enable_reload = false;

// These are required depending on tilize/untilize
uint32_t matmul_act_cb_id = 0;
uint32_t matmul_out_intermediate_cb_id = 24;
if constexpr (tilize_in) {
    // If we tilize, matmul doesn't consume original input,
    // it consumes what is produced by tilize
    matmul_act_cb_id = 24;
    matmul_out_intermediate_cb_id = 25; // Given 24 is no longer available, we use 25 instead
}

uint32_t reblock_cb_id = 26; // Only used if untilize is required
uint32_t matmul_out_cb_id = 16;

for (uint32_t block = 0U; block < num_blocks - 1; block++) {
  if constexpr (tilize_in) {
    tilize_activation(
        in0_subblock_h,
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        matmul_act_cb_id);
  }

  pack_block(
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_num_tiles,
        matmul_out_intermediate_cb_id);
}

// Last block
if constexpr (tilize_in) {
    tilize_activation(
        in0_subblock_h,
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        matmul_act_cb_id);
}

if constexpr (untilize_out) {
   pack_block_and_untilize(
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_num_tiles,
        out_subblock_h,
        out_block_w,
        matmul_out_intermediate_cb_id,
        reblock_cb_id
    );
} else {
    pack_block(
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_num_tiles,
        matmul_out_cb_id);
}


}
}
