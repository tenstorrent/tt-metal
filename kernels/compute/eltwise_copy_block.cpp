#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t block_num_tiles;
    std::int32_t num_blocks;
};
  
void compute_main(const hlk_args_t *args) {

    for(int block = 0; block < args->num_blocks; ++block) {
       acquire_dst(DstMode::Half);

       // Wait tiles on the input / copy to dst / pop from input
       cb_wait_front(CB::c_in0, args->block_num_tiles);
       for(int t = 0; t < args->block_num_tiles; ++t) {
           copy_tile(CB::c_in0, t, t);
       }
       cb_pop_front(CB::c_in0, args->block_num_tiles);

       // Reserve space in output / pack / push to output
       cb_reserve_back(CB::c_out0, args->block_num_tiles);
       for(int t = 0; t < args->block_num_tiles; ++t) {
            pack_tile(t, CB::c_out0);
       }
       cb_push_back(CB::c_out0, args->block_num_tiles);

       release_dst(DstMode::Half);
    }

}
