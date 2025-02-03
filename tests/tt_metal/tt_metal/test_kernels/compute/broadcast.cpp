// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

#ifndef BCAST_OP_INIT
    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
#else
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1);
    BCAST_OP_INIT(tt::CBIndex::c_0, tt::CBIndex::c_1);
#endif

    cb_wait_front(tt::CBIndex::c_1, onetile);
    cb_reserve_back(tt::CBIndex::c_16, onetile);
    acquire_dst();
    cb_wait_front(tt::CBIndex::c_0, onetile);

#ifndef BCAST_SPECIFIC
    BCAST_OP<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#else
    BCAST_OP(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#endif
    pack_tile(0, tt::CBIndex::c_16);

    cb_pop_front(tt::CBIndex::c_0, onetile);
    release_dst();
    cb_push_back(tt::CBIndex::c_16, onetile);
    cb_pop_front(tt::CBIndex::c_1, onetile);
}
}  // namespace NAMESPACE
