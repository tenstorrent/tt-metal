// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"

#ifndef BCAST_ROW_IDX
#define BCAST_ROW_IDX 0
#endif

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

#ifndef BCAST_OP_INIT
    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
#else
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    BCAST_OP_INIT(tt::CBIndex::c_0, tt::CBIndex::c_1);
#endif

    cb_wait_front(tt::CBIndex::c_1, onetile);
    cb_reserve_back(tt::CBIndex::c_16, onetile);
    acquire_dst();
    cb_wait_front(tt::CBIndex::c_0, onetile);

#ifndef BCAST_SPECIFIC
    // For template version, use compile-time check for ROW broadcast
    if constexpr (BCAST_DIM == BroadcastType::ROW) {
        BCAST_OP<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, BCAST_ROW_IDX);
    } else {
        BCAST_OP<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
    }
#else
// For specific function calls, check if BCAST_IS_ROW is defined
#ifdef BCAST_IS_ROW
    // Row broadcast functions have the bcast_row_idx parameter
    BCAST_OP(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, BCAST_ROW_IDX);
#else
    // Col and Scalar broadcast functions don't have the parameter
    BCAST_OP(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#endif
#endif

    pack_tile(0, tt::CBIndex::c_16);

    cb_pop_front(tt::CBIndex::c_0, onetile);
    release_dst();
    cb_push_back(tt::CBIndex::c_16, onetile);
    cb_pop_front(tt::CBIndex::c_1, onetile);
}
}  // namespace NAMESPACE
