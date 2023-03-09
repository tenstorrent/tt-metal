#include <cstdint>

#include "compute_hlk_api.h"

void compute_main() {

    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    for(int block = 0; block < num_blocks; ++block) {
       acquire_dst(DstMode::Half);

       // Wait tiles on the input / copy to dst / pop from input
       cb_wait_front(CB::c_in0, block_num_tiles);
       for(int t = 0; t < block_num_tiles; ++t) {
           copy_tile(CB::c_in0, t, t);
       }
       cb_pop_front(CB::c_in0, block_num_tiles);

       // Reserve space in output / pack / push to output
       cb_reserve_back(CB::c_out0, block_num_tiles);
       for(int t = 0; t < block_num_tiles; ++t) {
            pack_tile(t, CB::c_out0);
       }
       cb_push_back(CB::c_out0, block_num_tiles);

       release_dst(DstMode::Half);
    }

}
