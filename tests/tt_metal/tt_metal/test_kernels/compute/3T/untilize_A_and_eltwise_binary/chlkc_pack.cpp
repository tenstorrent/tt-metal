// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"
namespace NAMESPACE
{

void pack_main()
{
uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);
llk_pack_init();
llk_pack_hw_configure_disaggregated<false>(16);
llk_pack_dest_init<DstTileFaceLayout::RowMajor, false>();

for (uint32_t block = 0; block < per_core_num_blocks; block++) {
    for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
        llk_wait_for_free_tiles<false,false,false>(24, per_core_block_c_tiles);
        for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
            llk_packer_wait_for_math_done();
            llk_pack<false, false >(0,24);
            llk_pack_dest_section_done();
        }
        llk_push_tiles<false, false>(24, per_core_block_c_tiles);

        llk_wait_for_free_tiles<false,false,false>(16, per_core_block_c_tiles);
        for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
            llk_packer_wait_for_math_done();
            llk_pack<false, false >(0,16);
            llk_pack_dest_section_done();
        }
        llk_push_tiles<false, false>(16, per_core_block_c_tiles);
    }
}

}
}
