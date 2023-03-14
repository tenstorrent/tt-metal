#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"
namespace NAMESPACE
{

void pack_main()
{
uint32_t in0_block_w = get_compile_time_arg_val(0);
llk_pack_init();
llk_setup_outputs();
llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>();
llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>();
llk_pack_hw_configure_disaggregated<false>(16);
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

for (uint32_t block = 0U; block < num_blocks; block++) {
  bool last_out = block == num_blocks - 1U;
  llk_wait_for_free_tiles<false,false,false>(25,in0_block_num_tiles);
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    for (uint32_t i = 0U; i < in0_subblock_h; i++) {
      for (uint32_t j = 0U; j < in0_block_w; j++) {
        llk_packer_wait_for_math_done();
        llk_pack<false, SyncHalf, false >(0,25);
        llk_pack_dest_section_done<SyncHalf>();
        llk_push_tiles<false,false>(25,1);
      }
    }
  }
  for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
    for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
      llk_packer_wait_for_math_done();
// Compute output sub-block from in0_subblock x in1_subblock
      if (last_out) {
        llk_wait_for_free_tiles<false,false,false>(16,out_subblock_num_tiles);
        for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
          llk_pack<false, SyncHalf, false >(i,16);
        }
        llk_push_tiles<false,false>(16,out_subblock_num_tiles);
      }
       else {
        llk_wait_for_free_tiles<false,false,false>(24,out_subblock_num_tiles);
        for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
          llk_pack<false, SyncHalf, false >(i,24);
        }
        // mbox[0] = in0_subblock;
        llk_push_tiles<false,false>(24,out_subblock_num_tiles);
      }
      llk_pack_dest_section_done<SyncHalf>();
    }
  }
  if (spill)
    enable_reload = true;
}
}
}
