#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_dim;
};

void compute_main(const hlk_args_t *args) {
    // expands to hlk_relu_config(nullptr, 1); for relu only
    INIT_RELU
    for (int block_index = 0; block_index < args->per_core_block_cnt; block_index++) {
        cb_reserve_back(CB::c_out0, args->per_core_block_dim);
        for(int tile_index = 0; tile_index < args->per_core_block_dim; ++tile_index) {
            acquire_dst(DstMode::Half);

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(CB::c_in0, 1);

            copy_tile(CB::c_in0, 0, 0);
            // SFPU_OP expected to be defined via add_define as one of
            // exp_tile, gelu_tile, recip_tile
            SFPU_OP_AND_PACK
            // comes from add_define in kernel config
            // Also is epxected to include pack_tile(0, CB::c_out0); for non-relu
            // For relu it expects the hlk_pack_relu variant

            cb_pop_front(CB::c_in0, 1);

            release_dst(DstMode::Half);
        }
        cb_push_back(CB::c_out0, args->per_core_block_dim);
    }
    DEINIT_RELU
    // expands to hlk_relu_config(nullptr, 0); for relu only
}
