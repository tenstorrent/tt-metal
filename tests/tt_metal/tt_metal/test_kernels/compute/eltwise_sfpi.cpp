// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // expands to hlk_relu_config(nullptr, 1); for relu only

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    INIT_RELU
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(CB::c_in0, 1);

            copy_tile(CB::c_in0, 0, 0);
            // SFPU_OP expected to be defined via add_define as one of
            // exp_tile, gelu_tile, recip_tile. etc followed by pack_tile
            // (except for relu because the llk is fused for relu)
            // "sfpu_gelu(0); pack_tile(0, CB::c_out0);"

            SFPI_OP_AND_PACK
            // comes from add_define in kernel config
            // Also is epxected to include pack_tile(0, CB::c_out0); for non-relu
            // For relu it expects the hlk_pack_relu variant

            cb_pop_front(CB::c_in0, 1);

            release_dst();
        }
        cb_push_back(CB::c_out0, per_core_block_dim);
    }
    DEINIT_RELU
    // expands to hlk_relu_config(nullptr, 0); for relu only
}
}
