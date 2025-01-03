// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_compile_time_arg_val(0);
#ifndef SHORT_INIT
    transpose_wh_init(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    unary_op_init_common(tt::CBIndex::c_0);
    transpose_wh_init_short(tt::CBIndex::c_0);
#endif

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);

        acquire_dst();
        transpose_wh_tile(tt::CBIndex::c_0, 0, 0);
        pack_tile(0, tt::CBIndex::c_16);
        release_dst();

        cb_push_back(tt::CBIndex::c_16, 1);
        cb_pop_front(tt::CBIndex::c_0, 1);
    }
}
}  // namespace NAMESPACE
