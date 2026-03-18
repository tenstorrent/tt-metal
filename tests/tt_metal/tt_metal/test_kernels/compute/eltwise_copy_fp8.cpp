// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

#ifdef PACK_A_TO_B
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_In_data_format_RMW>(to_underlying(PACK_A_TO_B));
#endif
#ifdef PACK_B_TO_A
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(to_underlying(PACK_B_TO_A));
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Round_10b_mant_RMW>(1);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();

        cb0.wait_front(1);
        cb16.reserve_back(1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        pack_tile(0, tt::CBIndex::c_16);
        cb0.pop_front(1);
        cb16.push_back(1);

        release_dst();
    }
}
