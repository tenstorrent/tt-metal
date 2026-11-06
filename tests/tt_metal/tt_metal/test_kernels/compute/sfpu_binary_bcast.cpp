// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Compile-time parameters (set from host via defines):
//   FORM_VAL      : 0 = the whole tile, 1 = rows 0..15 (VectorMode::R), 2 = one 4-row band (sfpu_*_bcast_col_band)
//   BCAST_DIM_VAL : 0 = COL,  1 = ROW              -> ckernel::BroadcastType   (forms 0 and 1)
//   BINOP_VAL     : 0 = ADD,  1 = SUB,  2 = MUL    -> ckernel::EltwiseBinaryType (forms 0..2)
//   DATA_BAND_VAL, BCAST_BAND_VAL : the bands of form 2
//   NUM_OUT_TILES_VAL : 1 or 2 output tiles.
//
// DST layout: dst[0] = data tile (in place, the output), dst[1] = the broadcast tile.

#ifndef FORM_VAL
#define FORM_VAL 0
#endif
#ifndef BCAST_DIM_VAL
#define BCAST_DIM_VAL 0
#endif
#ifndef BINOP_VAL
#define BINOP_VAL 0
#endif
#ifndef DATA_BAND_VAL
#define DATA_BAND_VAL 0
#endif
#ifndef BCAST_BAND_VAL
#define BCAST_BAND_VAL 0
#endif
#ifndef NUM_OUT_TILES_VAL
#define NUM_OUT_TILES_VAL 1
#endif

namespace {

constexpr uint32_t kForm = FORM_VAL;
constexpr uint32_t kNumOutTiles = NUM_OUT_TILES_VAL;

constexpr uint32_t kDstData = 0;
constexpr uint32_t kDstBcast = 1;

constexpr ckernel::BroadcastType kBcastDim =
    (BCAST_DIM_VAL == 0) ? ckernel::BroadcastType::COL : ckernel::BroadcastType::ROW;

constexpr ckernel::EltwiseBinaryType kBcastOp = (BINOP_VAL == 0)   ? ckernel::EltwiseBinaryType::ELWADD
                                                : (BINOP_VAL == 1) ? ckernel::EltwiseBinaryType::ELWSUB
                                                                   : ckernel::EltwiseBinaryType::ELWMUL;

}  // namespace

void kernel_main() {
    constexpr int NUM_IN_TILES = 1;

    constexpr auto input_a_cb_id = tt::CBIndex::c_0;
    constexpr auto input_b_cb_id = tt::CBIndex::c_1;
    constexpr auto out_cb_id = tt::CBIndex::c_16;

    CircularBuffer input_a_cb(input_a_cb_id);
    CircularBuffer input_b_cb(input_b_cb_id);
    CircularBuffer out_cb(out_cb_id);

    compute_kernel_hw_startup(input_a_cb_id, out_cb_id);
    copy_init(input_a_cb_id);
    sfpu_bcast_init<kBcastDim>();

    input_a_cb.wait_front(NUM_IN_TILES);
    input_b_cb.wait_front(NUM_IN_TILES);
    out_cb.reserve_back(kNumOutTiles);

    tile_regs_acquire();

    copy_init(input_a_cb_id);
    copy_tile(input_a_cb_id, 0, kDstData);
    copy_tile(input_b_cb_id, 0, kDstBcast);

    if constexpr (kForm == 0) {
        sfpu_bcast<kBcastDim, kBcastOp>(kDstData, kDstBcast);
    } else if constexpr (kForm == 1) {
        sfpu_bcast<kBcastDim, kBcastOp>(kDstData, kDstBcast, ckernel::VectorMode::R);
    } else if constexpr (kForm == 2) {
        sfpu_bcast_col_band<kBcastOp>(kDstData, kDstBcast, DATA_BAND_VAL, BCAST_BAND_VAL);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(kDstData, out_cb_id);

    tile_regs_release();

    input_a_cb.pop_front(NUM_IN_TILES);
    input_b_cb.pop_front(NUM_IN_TILES);
    out_cb.push_back(kNumOutTiles);
}
