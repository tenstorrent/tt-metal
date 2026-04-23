// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

// Compile-time parameters (set from host via defines):
//   BCAST_DIM_VAL : 0 = BCAST_COL, 1 = BCAST_ROW         (ckernel::SfpuBcastDim)
//   BINOP_VAL     : 0 = ADD,       1 = SUB,    2 = MUL   (ckernel::SfpuBcastOp)
//
// DST layout used by this kernel:
//   dst[0] = data tile   (input A, also output written in-place)
//   dst[1] = broadcast tile (input B; col 0 read for COL, row 0 read for ROW)

namespace {

constexpr uint32_t kDstData = 0;
constexpr uint32_t kDstBcast = 1;

constexpr auto kBcastDim = static_cast<ckernel::SfpuBcastDim>(BCAST_DIM_VAL);
constexpr auto kBcastOp = static_cast<ckernel::SfpuBcastOp>(BINOP_VAL);

}  // namespace

void kernel_main() {
    constexpr int NUM_TILES = 1;

    constexpr auto cb_data = tt::CBIndex::c_0;
    constexpr auto cb_bcast = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    experimental::CircularBuffer cb0(cb_data);
    experimental::CircularBuffer cb1(cb_bcast);
    experimental::CircularBuffer cb16(cb_out);

    init_sfpu(cb_data, cb_out);
    sfpu_bcast_init<kBcastDim>();

    cb0.wait_front(NUM_TILES);
    cb1.wait_front(NUM_TILES);
    cb16.reserve_back(NUM_TILES);

    acquire_dst();

    copy_tile_to_dst_init_short(cb_data);
    copy_tile(cb_data, 0, kDstData);

    copy_tile_to_dst_init_short(cb_bcast);
    copy_tile(cb_bcast, 0, kDstBcast);

    sfpu_bcast<kBcastDim, kBcastOp>(kDstData, kDstBcast);

    pack_tile(kDstData, cb_out);

    release_dst();

    cb0.pop_front(NUM_TILES);
    cb1.pop_front(NUM_TILES);
    cb16.push_back(NUM_TILES);
}
