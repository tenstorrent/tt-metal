// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_unary_datacopy_api.h"

void math_main() {
    int __outer_loop_iter;
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(0)));
    llk_math_pack_sync_init<DST_ACCUM_MODE>();
    llk_math_hw_configure<DST_ACCUM_MODE>(0, 0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        llk_math_wait_for_dest_available();
        llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(0);
        llk_math_dest_section_done<DST_ACCUM_MODE>();
    }
}
#endif

#ifdef TRISC_PACK
#include "llk_pack_common.h"
#include "llk_pack.h"

void pack_main() {
    int __outer_loop_iter;
    llk_pack_init();
    llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(16);
    llk_pack_dest_init<DST_ACCUM_MODE, false>();
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        llk_packer_wait_for_math_done();
        llk_wait_for_free_tiles<false, false, false>(16, 1);
        llk_pack<DST_ACCUM_MODE, false, false>(0, 16);
        llk_push_tiles<false, false>(16, 1);
        llk_pack_dest_section_done<DST_ACCUM_MODE>();
    }
}
#endif

#ifdef TRISC_UNPACK
void unpack_main() {
    int __outer_loop_iter;
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>()));
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(0)));
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        llk_wait_tiles(0, 1);
        llk_unpack_A(0, 0);
        llk_pop_tiles(0, 1);
    }
}
#endif

}  // namespace NAMESPACE
