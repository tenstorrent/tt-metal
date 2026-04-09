// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/dataflow_buffer.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr bool use_dfbs = get_compile_time_arg_val(1) == 1;

    if constexpr (use_dfbs) {
        unary_op_init_common(0, 1);
    } else {
        unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    }

#ifdef PACK_RELU
#ifdef ARCH_QUASAR
    pack_relu_config(ReluConfig::from_packed(get_arg_val<uint32_t>(0)));
#else
    pack_relu_config(get_arg_val<uint32_t>(0));
#endif
#endif

    if constexpr (use_dfbs) {
        experimental::DataflowBuffer dfb_in(0);
        experimental::DataflowBuffer dfb_out(1);
        for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
            acquire_dst();

            dfb_in.wait_front(1);
            dfb_out.reserve_back(1);
            copy_tile(dfb_in.get_id(), 0, 0);
            pack_tile(0, dfb_out.get_id());
            dfb_in.pop_front(1);
            dfb_out.push_back(1);

            release_dst();
        }
    } else {
        experimental::CircularBuffer cb0(tt::CBIndex::c_0);
        experimental::CircularBuffer cb16(tt::CBIndex::c_16);
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
}
