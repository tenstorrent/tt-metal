// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint_tensix.h"

//#include "debug/dprint.h"
inline void tilizeA_B_binary_init(uint32_t icb0, uint32_t icb1, uint32_t block, uint32_t ocb = 16, uint32_t num_faces = 4, uint32_t face_r_dim = 16) {
    UNPACK(( llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_tilizeA_B_init<true, true>(icb0, icb1, block, num_faces, face_r_dim, face_r_dim) ));

    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));
    MATH(( llk_math_hw_configure_disaggregated(icb0, icb1) ));
    MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>(0/*transpose*/, 0/*acc_to_dest*/) ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>(ocb) ));
}

inline void add_tiles_math(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH(( llk_math_eltwise_binary<ELWADD, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(icb0, icb1, idst) ));
}

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    tilizeA_B_binary_init(tt::CB::c_in0, tt::CB::c_in1, per_core_block_tile_cnt, tt::CB::c_out0);

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_wait_front(tt::CB::c_in1, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);
        unpack_tilizeA_B_block(tt::CB::c_in0, tt::CB::c_in1, per_core_block_tile_cnt, b);

        for(uint i=0; i<per_core_block_tile_cnt; ++i) {
            acquire_dst();
            add_tiles_math(tt::CB::c_in0, tt::CB::c_in1, i, i, 0);
            // dprint_tensix_dest_reg(0);
            pack_tile(0, tt::CB::c_out0);
            release_dst();
        }

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in1, per_core_block_tile_cnt);
    }
}
}
