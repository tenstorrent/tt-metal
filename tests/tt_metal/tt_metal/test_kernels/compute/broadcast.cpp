// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

#ifndef BCAST_ROW_IDX
#define BCAST_ROW_IDX 0
#endif

void kernel_main() {
    constexpr uint32_t onetile = 1;

#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_src_a_id = get_compile_time_arg_val(0);
    constexpr uint32_t dfb_src_b_id = get_compile_time_arg_val(1);
    constexpr uint32_t dfb_dst_id = get_compile_time_arg_val(2);
    experimental::DataflowBuffer dfb_src_a(dfb_src_a_id);
    experimental::DataflowBuffer dfb_src_b(dfb_src_b_id);
    experimental::DataflowBuffer dfb_dst(dfb_dst_id);
#endif

#ifndef BCAST_OP_INIT
#ifdef ARCH_QUASAR
    init_bcast<BCAST_LLKOP, BCAST_DIM>(dfb_src_a.get_id(), dfb_src_b.get_id(), dfb_dst.get_id());
#else
    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
#endif
#else
#ifdef ARCH_QUASAR
    binary_op_init_common(dfb_src_a.get_id(), dfb_src_b.get_id(), dfb_dst.get_id());
    BCAST_OP_INIT(dfb_src_a.get_id(), dfb_src_b.get_id());
#else
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    BCAST_OP_INIT(tt::CBIndex::c_0, tt::CBIndex::c_1);
#endif
#endif

#ifdef ARCH_QUASAR
    dfb_src_b.wait_front(onetile);
    dfb_dst.reserve_back(onetile);
    acquire_dst();
    dfb_src_a.wait_front(onetile);
#else
    experimental::CircularBuffer cb1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);

    cb1.wait_front(onetile);
    cb16.reserve_back(onetile);
    acquire_dst();
    cb0.wait_front(onetile);
#endif

#ifndef BCAST_SPECIFIC
    // For template version, use compile-time check for ROW broadcast
    if constexpr (BCAST_DIM == BroadcastType::ROW) {
#ifdef ARCH_QUASAR
        BCAST_OP<BCAST_DIM>(dfb_src_a.get_id(), dfb_src_b.get_id(), 0, 0, 0, BCAST_ROW_IDX);
#else
        BCAST_OP<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, BCAST_ROW_IDX);
#endif
    } else {
#ifdef ARCH_QUASAR
        BCAST_OP<BCAST_DIM>(dfb_src_a.get_id(), dfb_src_b.get_id(), 0, 0, 0);
#else
        BCAST_OP<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#endif
    }
#else
// For specific function calls, check if BCAST_IS_ROW is defined
#ifdef BCAST_IS_ROW
    // Row broadcast functions have the bcast_row_idx parameter
#ifdef ARCH_QUASAR
    BCAST_OP(dfb_src_a.get_id(), dfb_src_b.get_id(), 0, 0, 0, BCAST_ROW_IDX);
#else
    BCAST_OP(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, BCAST_ROW_IDX);
#endif
#else
    // Col and Scalar broadcast functions don't have the parameter
#ifdef ARCH_QUASAR
    BCAST_OP(dfb_src_a.get_id(), dfb_src_b.get_id(), 0, 0, 0);
#else
    BCAST_OP(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#endif
#endif
#endif

#ifdef ARCH_QUASAR
    pack_tile(0, dfb_dst.get_id());
    dfb_src_a.pop_front(onetile);
    release_dst();
    dfb_dst.push_back(onetile);
    dfb_src_b.pop_front(onetile);
#else
    pack_tile(0, tt::CBIndex::c_16);
    cb0.pop_front(onetile);
    release_dst();
    cb16.push_back(onetile);
    cb1.pop_front(onetile);
#endif
}
