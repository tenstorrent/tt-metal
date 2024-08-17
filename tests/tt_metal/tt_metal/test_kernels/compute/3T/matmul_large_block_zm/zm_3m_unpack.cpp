// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#include "llk_unpack_A_api.h"
#include "llk_unpack_AB_matmul_api.h"
namespace NAMESPACE
{

inline void tilize_activation(uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks) {
    // Tilize block code
    llk_unpack_tilize_init(0, in0_block_w);
    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t i = 0U; i < in0_subblock_h; i++) {
            llk_wait_tiles(0, in0_block_w); // These "tiles" are actually not real tiles
            llk_unpack_tilize_(0,in0_block_w);
            llk_pop_tiles(0,in0_block_w); // Pop the original untilized inputs
        }
    }
    llk_unpack_tilize_uninit(0);
}


inline __attribute__((always_inline))
void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    uint32_t interm_cb_id,
    uint32_t reblock_cb_id) {

    // Wait for a row of subblocks such that the total width matches
    // the out block width. Must wait for a whole row of subblocks to arrive
    // before we can proceed.
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles,  num_out_subblocks_in_col);
    llk_wait_tiles(interm_cb_id, num_tiles_in_row_of_subblocks);

    int within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        int block_offset = 0;

        llk_unpack_A_init<BroadcastType::NONE, false, false>();
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                llk_unpack_A(interm_cb_id, tile_index);
            }
            block_offset += out_subblock_num_tiles;
        }

        // Since our reblock CB can only fit one row of
        // tiles, we need to immediately untilize to
        // consume this row
        llk_wait_tiles(reblock_cb_id, out_block_w);
        /*
        for (uint32_t i = 0; i < out_block_w; i++) {
            llk_unpack_A(reblock_cb_id, i);
        }
        */

            llk_unpack_untilize_init(reblock_cb_id);
            llk_unpack_untilize_<true>(reblock_cb_id, out_block_w);
            llk_unpack_untilize_<false>(reblock_cb_id, out_block_w);
            llk_unpack_untilize_uninit(reblock_cb_id);

        llk_pop_tiles(reblock_cb_id, out_block_w);

        within_block_index += out_subblock_w;
    }
    llk_pop_tiles(interm_cb_id, num_tiles_in_row_of_subblocks);
}

inline void unpack_for_matmul_output_row(
    uint32_t in1_num_subblocks,
    bool enable_reload,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t in0_block_w,
    uint32_t in0_index_subblock_offset,
    uint32_t in1_per_core_w,
    uint32_t matmul_act_cb_id,
    uint32_t matmul_out_intermediate_cb_id) {

    uint32_t in1_index_subblock_offset = 0;
    for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
      if (enable_reload) {
        llk_unpack_A_init<BroadcastType::NONE, false, false>();
        llk_wait_tiles(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
          llk_unpack_A(matmul_out_intermediate_cb_id, i);
        }
        llk_pop_tiles(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
      }

      llk_unpack_AB_matmul_init(0);
      int dst_index = 0;
      int in0_index_h_offset = 0;
      for (uint32_t h = 0U; h < out_subblock_h; h++) {
        for (uint32_t w = 0U; w < out_subblock_w; w++) {
          int in1_index_inner_dim_offset = 0;
          for (uint32_t inner_dim = 0U; inner_dim < in0_block_w; inner_dim++) {
            int in0_index = ((in0_index_subblock_offset + in0_index_h_offset) + inner_dim);
            int in1_index = ((in1_index_subblock_offset + in1_index_inner_dim_offset) + w);
            llk_unpack_AB_matmul(matmul_act_cb_id, 1, in0_index, in1_index);
            in1_index_inner_dim_offset += in1_per_core_w;
          }
          dst_index++;
        }
        in0_index_h_offset += in0_block_w;
      }
      in1_index_subblock_offset += out_subblock_w;
    }
}

void unpack_main()
{
uint32_t in0_block_w = get_compile_time_arg_val(0);
llk_unpack_AB_matmul_init(0);
// inner block size in tiles
uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
// outer row block size (in inner row blocks)
uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
// out_subblock_h*in0_block_w*in0_num_subblocks;
uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);

uint32_t in0_subblock_h = get_compile_time_arg_val(4);

// out_subblock_h*in0_block_w
uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
// outer column block size (in inner column blocks)
uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
//out_subblock_w*in0_block_w* in1_num_subblocks;
uint32_t in1_per_core_w = get_compile_time_arg_val(7);
// out_subblock_w*in1_num_subblocks
constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
// outer inner dim (in inner dim blocks)
uint32_t out_subblock_h = get_compile_time_arg_val(9);
// inner row block size in tiles
uint32_t out_subblock_w = get_compile_time_arg_val(10);
// inner column block size in tiles
uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);

uint32_t out_block_w = in1_per_core_w;

// If true, this assumes data coming in RM
constexpr bool tilize_in = get_compile_time_arg_val(12);

// If true, this assumes consumer wants data RM
constexpr bool untilize_out = get_compile_time_arg_val(13);


// These are required depending on tilize/untilize
uint32_t matmul_act_cb_id = 0;
uint32_t matmul_out_intermediate_cb_id = 24;
if constexpr (tilize_in) {
    // If we tilize, matmul doesn't consume original input,
    // it consumes what is produced by tilize
    matmul_act_cb_id = 24;

    matmul_out_intermediate_cb_id = 25; // Given 24 is no longer available, we use 25 instead
}

llk_unpack_AB_matmul_hw_configure_disaggregated(0,1,0);

uint32_t reblock_cb_id = 26;

constexpr bool spill = num_blocks > 1U;
bool enable_reload = false;
for (uint32_t block = 0U; block < num_blocks; block++) {
  bool last_out = block == num_blocks - 1U;

  if constexpr (tilize_in) {
    tilize_activation(in0_subblock_h, in0_block_w, in0_num_subblocks);
  } else {
    llk_wait_tiles(matmul_act_cb_id, in0_block_num_tiles);
  }

  // Wait on weight tiles
  llk_wait_tiles(1, in1_block_num_tiles);
  int in0_index_subblock_offset = 0;
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    unpack_for_matmul_output_row(
        in1_num_subblocks,
        enable_reload,
        out_subblock_num_tiles,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        in0_index_subblock_offset,
        in1_per_core_w,
        matmul_act_cb_id,
        matmul_out_intermediate_cb_id);

    if constexpr (untilize_out) {
        if (last_out) {
            reblock_and_untilize(
                in1_num_subblocks,
                out_subblock_num_tiles,
                out_subblock_h,
                out_subblock_w,
                out_block_w,
                matmul_out_intermediate_cb_id,
                reblock_cb_id);
        }
    }

    in0_index_subblock_offset += in0_subblock_num_tiles;
  }

  // Need to do a reblock datacopy
  if constexpr (spill) {
    enable_reload = true;
  }

  llk_pop_tiles(matmul_act_cb_id, in0_block_num_tiles);
  llk_pop_tiles(1, in1_block_num_tiles);
}

}
}
