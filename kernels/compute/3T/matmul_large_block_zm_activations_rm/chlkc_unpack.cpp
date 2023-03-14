#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB_matmul.h"
namespace NAMESPACE
{

void unpack_main()
{
uint32_t in0_block_w = get_compile_time_arg_val(0);
llk_setup_operands();
llk_unpack_AB_matmul_init(0);
llk_unpack_AB_matmul_hw_configure_disaggregated(25,1,0);
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
volatile uint32_t* mbox = reinterpret_cast<uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);
volatile uint32_t* mbox2 = reinterpret_cast<uint32_t*>(l1_mem::address_map::TRISC1_DEBUG_BUFFER_BASE);
// mbox2[0] = num_blocks;
for (uint32_t block = 0U; block < num_blocks; block++) {
  bool last_out = block == num_blocks - 1U;

  // Tilize block code
  llk_unpack_tilize_init(0, in0_block_w);
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    for (uint32_t i = 0U; i < in0_subblock_h; i++) {
        llk_wait_tiles(0, in0_block_w); // These "tiles" are actually not real tiles
        llk_unpack_tilize_(0,in0_block_w);
        llk_pop_tiles(0,in0_block_w); // Pop the original untilized inputs
    }

  }
  llk_unpack_tilize_uninit();

  // Wait on weight tiles
  llk_wait_tiles(1,in1_block_num_tiles);
  int in0_index_subblock_offset = 0;
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    int in1_index_subblock_offset = 0;
    for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
      if (enable_reload) {
        llk_unpack_A_init<BroadcastType::NONE, false, false>();
        llk_wait_tiles(24,out_subblock_num_tiles);
        for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
          llk_unpack_A(24,i);
        }
        llk_pop_tiles(24,out_subblock_num_tiles);
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
            llk_unpack_AB_matmul(25,1,in0_index,in1_index);
            in1_index_inner_dim_offset += in1_per_core_w;
          }
          dst_index++;
        }
        in0_index_h_offset += in0_block_w;
      }
      in1_index_subblock_offset += out_subblock_w;
    }
    in0_index_subblock_offset += in0_subblock_num_tiles;
  }
  if (spill)
    enable_reload = true;
  llk_pop_tiles(25,in0_block_num_tiles);
  llk_pop_tiles(1,in1_block_num_tiles);
}
}
}
