#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"
namespace NAMESPACE
{

void math_main()
{
uint32_t in0_block_w = get_compile_time_arg_val(0);
llk_math_pack_sync_init<SyncHalf>();

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
uint32_t num_blocks = get_compile_time_arg_val(8);
// outer inner dim (in inner dim blocks)
uint32_t out_subblock_h = get_compile_time_arg_val(9);
// inner row block size in tiles
uint32_t out_subblock_w = get_compile_time_arg_val(10);
// inner column block size in tiles
uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);
// out_subblock_h * out_subblock_w;
// Needed for tilize
bool spill = num_blocks > 1U;
bool enable_reload = false;

for (uint32_t block = 0U; block < num_blocks; block++) {
  bool last_out = block == num_blocks - 1U;
  llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    for (uint32_t i = 0U; i < in0_subblock_h; i++) {
      for (uint32_t j = 0U; j < in0_block_w; j++) {
        llk_math_wait_for_dest_available<SyncHalf>();
        llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0);
        llk_math_dest_section_done<SyncHalf>();
      }
    }
  }
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {

      llk_math_wait_for_dest_available<SyncHalf>();
      if (enable_reload) {
        llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
        for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
          llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(i);
        }
      }
      llk_math_matmul_init<MATH_FIDELITY>(0);
// Compute output sub-block from in0_subblock x in1_subblock
      int dst_index = 0;
      for (uint32_t h = 0U; h < out_subblock_h; h++) {
        for (uint32_t w = 0U; w < out_subblock_w; w++) {
          for (uint32_t inner_dim = 0U; inner_dim < in0_block_w; inner_dim++) {
            llk_math_matmul<MATH_FIDELITY>(dst_index);
          }
          dst_index++;
        }
      }

      llk_math_dest_section_done<SyncHalf>();
    }
  }
  if (spill)
    enable_reload = true;
}
}
}
