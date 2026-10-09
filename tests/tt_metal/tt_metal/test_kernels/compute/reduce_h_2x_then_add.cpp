// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"

// One-tile column (H) SUM reduce of in_data by in_scaler, then an ordinary add of the same two tiles.
// With both inputs MxFp4 the reduce runs on the Quasar 2x-packed src-register format, which reduce_init
// selects by overriding both unpackers' OUT_DATA_FORMAT and the ALU src formats. The add that follows
// unpacks the same buffers through UNP_A and UNP_B on the op-agnostic (Float16_b) formats, so it is only
// correct if reduce_uninit restored both unpackers and the ALU. Output: tile 0 = reduce, tile 1 = add.
void kernel_main() {
    constexpr std::uint32_t onetile = 1;

    DataflowBuffer dfb_in(dfb::in_data);
    DataflowBuffer dfb_in_scaler(dfb::in_scaler);
    DataflowBuffer dfb_out(dfb::out);
    compute_kernel_hw_startup(dfb::in_data, dfb::in_scaler, dfb::out);

    dfb_in.wait_front(onetile);
    dfb_in_scaler.wait_front(onetile);

    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(dfb::in_data, dfb::in_scaler, dfb::out);
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(dfb::in_data, dfb::in_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(onetile);
    pack_tile(0, dfb::out);
    dfb_out.push_back(onetile);
    tile_regs_release();
    reduce_uninit(dfb::in_data);

    add_init(dfb::in_data, dfb::in_scaler);
    tile_regs_acquire();
    add_tiles(dfb::in_data, dfb::in_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(onetile);
    pack_tile(0, dfb::out);
    dfb_out.push_back(onetile);
    tile_regs_release();

    dfb_in.pop_front(onetile);
    dfb_in_scaler.pop_front(onetile);
}
