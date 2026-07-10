// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

#ifndef BCAST_ROW_IDX
#define BCAST_ROW_IDX 0
#endif

void kernel_main() {
    constexpr uint32_t onetile = 1;

#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_in0_id = get_compile_time_arg_val(0);
    constexpr uint32_t dfb_in1_id = get_compile_time_arg_val(1);
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(2);
    DataflowBuffer dfb0(dfb_in0_id);
    DataflowBuffer dfb1(dfb_in1_id);
    DataflowBuffer dfb_out(dfb_out_id);
    const uint32_t icb0 = dfb0.get_id();
    const uint32_t icb1 = dfb1.get_id();
    const uint32_t ocb = dfb_out.get_id();
#else
    constexpr uint32_t icb0 = tt::CBIndex::c_0;
    constexpr uint32_t icb1 = tt::CBIndex::c_1;
    constexpr uint32_t ocb = tt::CBIndex::c_16;
    CircularBuffer cb1(icb1);
    CircularBuffer cb16(ocb);
    CircularBuffer cb0(icb0);
#endif

#ifndef BCAST_OP_INIT
    init_bcast<BCAST_LLKOP, BCAST_DIM>(icb0, icb1, ocb);
#else
    binary_op_init_common(icb0, icb1, ocb);
    BCAST_OP_INIT(icb0, icb1);
#endif

#ifdef ARCH_QUASAR
    dfb1.wait_front(onetile);
    dfb_out.reserve_back(onetile);
    tile_regs_acquire();
    dfb0.wait_front(onetile);
#else
    cb1.wait_front(onetile);
    cb16.reserve_back(onetile);
    tile_regs_acquire();
    cb0.wait_front(onetile);
#endif

#ifndef BCAST_SPECIFIC
    // For template version, use compile-time check for ROW broadcast
    if constexpr (BCAST_DIM == BroadcastType::ROW) {
        BCAST_OP<BCAST_DIM>(icb0, icb1, 0, 0, 0, BCAST_ROW_IDX);
    } else {
        BCAST_OP<BCAST_DIM>(icb0, icb1, 0, 0, 0);
    }
#else
// For specific function calls, check if BCAST_IS_ROW is defined
#ifdef BCAST_IS_ROW
    // Row broadcast functions have the bcast_row_idx parameter
    BCAST_OP(icb0, icb1, 0, 0, 0, BCAST_ROW_IDX);
#else
    // Col and Scalar broadcast functions don't have the parameter
    BCAST_OP(icb0, icb1, 0, 0, 0);
#endif
#endif

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, ocb);

#ifdef ARCH_QUASAR
    dfb0.pop_front(onetile);
    tile_regs_release();
    dfb_out.push_back(onetile);
    dfb1.pop_front(onetile);
#else
    cb0.pop_front(onetile);
    tile_regs_release();
    cb16.push_back(onetile);
    cb1.pop_front(onetile);
#endif
}
