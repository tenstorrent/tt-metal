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
        init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CB::c_in0, tt::CB::c_in1);
    #else
        binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1);
        BCAST_OP_INIT(tt::CB::c_in0, tt::CB::c_in1);
    #endif

    cb_wait_front(tt::CB::c_in1, onetile);
    cb_reserve_back(tt::CB::c_out0, onetile);
    acquire_dst(tt::DstMode::Half);
    cb_wait_front(tt::CB::c_in0, onetile);

    #ifndef BCAST_SPECIFIC
        BCAST_OP<BCAST_DIM>(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
    #else
        BCAST_OP(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
    #endif
    pack_tile(0, tt::CB::c_out0);

    cb_pop_front(tt::CB::c_in0, onetile);
    release_dst(tt::DstMode::Half);
    cb_push_back(tt::CB::c_out0, onetile);
    cb_pop_front(tt::CB::c_in1, onetile);
}
} // NAMESPACE
