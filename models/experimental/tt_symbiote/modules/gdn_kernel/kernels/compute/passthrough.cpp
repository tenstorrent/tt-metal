// Minimal passthrough compute kernel: just copy input CB to output CB
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    cb_wait_front(cb_in, num_tiles);
    cb_reserve_back(cb_out, num_tiles);

    copy_tile_to_dst_init_short(cb_in);
    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        copy_tile(cb_in, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, t);
        tile_regs_release();
    }

    cb_push_back(cb_out, num_tiles);
    cb_pop_front(cb_in, num_tiles);
}
