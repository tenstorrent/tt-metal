// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_untilize_api.h"

namespace NAMESPACE
{

void unpack_main()
{
uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(0,1);
// llk_unpack_untilize_hw_configure_disaggregated(0);

// llk_unpack_untilize_init(0);
for (uint32_t block = 0U; block < per_core_num_blocks; ++block) {
  for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
    llk_unpack_untilize_init(0);
    llk_wait_tiles(0, per_core_block_c_tiles);
    llk_unpack_untilize_<true>(0, per_core_block_c_tiles);
    llk_unpack_untilize_<false>(0, per_core_block_c_tiles);
    llk_unpack_untilize_uninit(0);
    llk_pop_tiles(0, per_core_block_c_tiles);
    llk_pop_tiles(1, per_core_block_c_tiles);

    llk_unpack_AB_init<BroadcastType::NONE>();
    for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
        llk_wait_tiles(24, 1);
        llk_wait_tiles(1, 1);
        llk_unpack_AB(24, 1, 0, 0);
        llk_pop_tiles(24, 1);
        llk_pop_tiles(1, 1);
    }
  }
}
}
}
