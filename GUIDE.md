```
#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1); // should be <= 8 in this kernel

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 =  tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init();

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // wait for a block of tiles in each of input CBs
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);

        tile_regs_acquire(); // acquire 8 tile registers
        // add a block of tiles
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            add_tiles(cb_in0, cb_in1, i, i, i);
        }
        tile_regs_commit(); // signal the packer 

        tile_regs_wait(); // packer waits here
        // pack a block of tiles
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            pack_tile(i, cb_out0);
        }
        tile_regs_release(); // packer releases

        // pop a block of tiles from each of input CBs 
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);

        // push a block of tiles to output CB
        cb_push_back(cb_out0, per_core_block_size);
    }

}
}
```



<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/6ea0cefc-6109-4579-8470-7a620f45b314">


<img width="1176" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/d3c89155-6e4d-49cb-a95c-85654ac29e7d">


<img width="1171" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/73039d17-3bce-4ff5-b797-da1aa9b147c4">
