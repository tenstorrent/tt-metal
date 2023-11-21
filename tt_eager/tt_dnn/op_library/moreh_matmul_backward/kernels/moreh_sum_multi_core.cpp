// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    const uint32_t num_src_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_dst_tiles = get_arg_val<uint32_t>(1);
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_zero = tt::CB::c_in1;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr auto cb_out = tt::CB::c_out0;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr int onetile = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1);
    cb_wait_front(cb_zero, onetile);

    for (uint32_t i = 0; i < num_dst_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_src_tiles; ++j) {
            bool last_out = (j == num_src_tiles - 1);
            uint32_t cb_src2 = (enable_reload) ? (cb_intermed0) : (cb_zero);

            ACQ();
            cb_wait_front(cb_in0, onetile);
            if (enable_reload) {
                cb_wait_front(cb_intermed0, onetile);
            }
            add_tiles_init();
            add_tiles(cb_in0, cb_src2, 0, 0, 0);

            cb_pop_front(cb_in0, onetile);
            if (enable_reload)
                cb_pop_front(cb_intermed0, onetile);

            if (last_out) {
                cb_reserve_back(cb_out, onetile);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, onetile);

            } else {
                cb_reserve_back(cb_intermed0, onetile);
                pack_tile(0, cb_intermed0);
                cb_push_back(cb_intermed0, onetile);
            }
            REL();
            enable_reload = true;
        }
    }
}
}  // namespace NAMESPACE
