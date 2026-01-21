// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "experimental/circular_buffer.h"

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

    experimental::CircularBuffer cb1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);

    cb1.wait_front(onetile);
    cb16.reserve_back(onetile);
    acquire_dst();
    cb0.wait_front(onetile);

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

    cb0.pop_front(onetile);
    release_dst();
    cb16.push_back(onetile);
    cb1.pop_front(onetile);
}
}  // namespace NAMESPACE
