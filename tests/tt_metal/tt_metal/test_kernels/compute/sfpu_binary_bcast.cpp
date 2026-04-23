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

    constexpr auto input_a_cb_id = tt::CBIndex::c_0;
    constexpr auto input_b_cb_id = tt::CBIndex::c_1;
    constexpr auto out_cb_id = tt::CBIndex::c_16;

    experimental::CircularBuffer input_a_cb(input_a_cb_id);
    experimental::CircularBuffer input_b_cb(input_b_cb_id);
    experimental::CircularBuffer out_cb(out_cb_id);

    init_sfpu(input_a_cb_id, out_cb_id);
    sfpu_bcast_init<kBcastDim>();

    input_a_cb.wait_front(NUM_TILES);
    input_b_cb.wait_front(NUM_TILES);
    out_cb.reserve_back(NUM_TILES);

    acquire_dst();

    copy_tile_to_dst_init_short(input_a_cb_id);
    copy_tile(input_a_cb_id, 0, kDstData);
    copy_tile(input_b_cb_id, 0, kDstBcast);

    sfpu_bcast<kBcastDim, kBcastOp>(kDstData, kDstBcast);

    pack_tile(kDstData, out_cb_id);

    release_dst();

    input_a_cb.pop_front(NUM_TILES);
    input_b_cb.pop_front(NUM_TILES);
    out_cb.push_back(NUM_TILES);
}
