// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

template <typename T>
inline T get_next_arg_val()
{
    static int arg_idx = 0;
    return get_arg_val<T> (arg_idx++);
}

namespace NAMESPACE {
void MAIN {
    const auto num_output_tiles = get_next_arg_val<uint32_t>();
    const auto n_need_bcast = get_next_arg_val<uint32_t>();
    const auto c_need_bcast = get_next_arg_val<uint32_t>();
    const auto ht_need_bcast = get_next_arg_val<uint32_t>();
    const auto wt_need_bcast = get_next_arg_val<uint32_t>();

    constexpr auto cb_in0 = tt::CB::c_in0;  // input
    constexpr auto cb_in1 = tt::CB::c_in1;  // zero tile
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1);
    cb_wait_front(cb_in1, onetile);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        ACQ();
        cb_wait_front(cb_in0, onetile);
        if (ht_need_bcast && wt_need_bcast) {
            add_bcast_scalar_init_short();
            add_tiles_bcast_scalar(cb_in1, cb_in0, 0, 0, dst0);
        } else if (ht_need_bcast) {
            add_bcast_rows_init_short();
            add_tiles_bcast_rows(cb_in1, cb_in0, 0, 0, dst0);
        } else if (wt_need_bcast) {
            add_bcast_cols_init_short();
            add_tiles_bcast_cols(cb_in1, cb_in0, 0, 0, dst0);
        } else {
            copy_tile_init();
            copy_tile(cb_in0, 0, dst0);
        }
        cb_reserve_back(cb_out0, onetile);
        pack_tile(dst0, cb_out0);
        cb_push_back(cb_out0, onetile);
        cb_pop_front(cb_in0, onetile);
        REL();
    }
    cb_pop_front(cb_in1, onetile);
}
}  // namespace NAMESPACE
