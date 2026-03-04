// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb_in(0);
    experimental::DataflowBuffer dfb_out(1);
    unary_op_init_common(dfb_in.get_id(), dfb_out.get_id());
#else
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();

#ifdef ARCH_QUASAR
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        pack_tile(0, dfb_out.get_id());
        dfb_in.pop_front(1);
        dfb_out.push_back(1);
#else
        // Pop tile after tile, copy to DST and pack
        cb0.wait_front(1);
        cb16.reserve_back(1);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
#endif

        release_dst();
    }
}
