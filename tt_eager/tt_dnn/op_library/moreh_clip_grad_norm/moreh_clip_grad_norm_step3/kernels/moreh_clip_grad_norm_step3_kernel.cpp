// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    const auto num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_x = tt::CB::c_in0;                  // input
    constexpr auto cb_clip_coef_clamped = tt::CB::c_in1;  // clip_coef_clamped

    constexpr auto cb_y = tt::CB::c_out0;  // output

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_x, cb_clip_coef_clamped);

    cb_wait_front(cb_clip_coef_clamped, onetile);  // comes from the reader

    // Compute cb_y
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        ACQ();
        cb_wait_front(cb_x, onetile);  // comes from the reader
        cb_reserve_back(cb_y, onetile);

        mul_tiles_bcast_scalar_init_short();
        mul_tiles_bcast_scalar(cb_x, cb_clip_coef_clamped, 0, 0, dst0);

        pack_tile(dst0, cb_y);

        cb_pop_front(cb_x, onetile);
        cb_push_back(cb_y, onetile);
        REL();
    }

    cb_pop_front(cb_clip_coef_clamped, onetile);
}  // void MAIN
}  // namespace NAMESPACE
