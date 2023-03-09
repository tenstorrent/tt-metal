#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
    std::int32_t num_tiles_c;
};

void compute_main(const hlk_args_t *args) {

    hlk_add_tile_init_once(nullptr);
    for(int block = 0; block < args->per_core_block_cnt; ++block) {

        cb_reserve_back(CB::c_out0, args->per_core_block_size);
        cb_reserve_back(CB::c_intermed0, args->per_core_block_size);

        // Tilize A
        cb_wait_front(CB::c_in0, args->per_core_block_size);
        hlk_tilize_and_copy_to_dst_init_short(nullptr);
        for (int t = 0; t < args->per_core_block_size; ++t) {
            acquire_dst(DstMode::Half);

            // The tilizer will tilize the whole block, whereas
            // math will copy tile by tile
            tilize_and_copy(0, t, 0, args->num_tiles_c);
            pack_tile(0, CB::c_intermed0);
            cb_pop_front(CB::c_in0, 1);
            cb_push_back(CB::c_intermed0, 1);
            release_dst(DstMode::Half);
        }

        cb_wait_front(CB::c_intermed0, args->per_core_block_size);
        for(int t = 0; t < args->per_core_block_size; ++t)
        {
            acquire_dst(DstMode::Half);

            // Binary add
            hlk_add_tile_init_short(nullptr);
            acquire_dst(DstMode::Half);
            cb_wait_front(CB::c_intermed0, 1);
            cb_wait_front(CB::c_in1, 1);

            ELTWISE_OP(CB::c_intermed0, CB::c_in1, 0, 0, 0);
            pack_tile(0, CB::c_out0);

            cb_pop_front(CB::c_intermed0, 1);
            cb_pop_front(CB::c_in1, 1);
            release_dst(DstMode::Half);
        }

        cb_push_back(CB::c_out0, args->per_core_block_size);
    }
}
