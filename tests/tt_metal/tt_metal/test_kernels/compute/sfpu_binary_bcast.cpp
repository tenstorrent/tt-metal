// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Compile-time parameters (set from host via defines):
//   FORM_VAL      : 0 = the whole tile, 1 = rows 0..15 (VectorMode::R), 2 = one 4-row band (sfpu_*_bcast_col_band),
//                   3 = the parked scalar (sfpu_bcast_scalar_load / sfpu_mul_bcast_scalar)
//   BCAST_DIM_VAL : 0 = COL,  1 = ROW              -> ckernel::BroadcastType   (forms 0 and 1)
//   BINOP_VAL     : 0 = ADD,  1 = SUB,  2 = MUL    -> ckernel::EltwiseBinaryType (forms 0..2)
//   DATA_BAND_VAL, BCAST_BAND_VAL : the bands of form 2
//   SCALAR_ITER_VAL (1..8), SCALAR_MODE_VAL (0 = RC, 1 = R, 2 = C), DST_DATA_VAL, DST_SCALAR_VAL, DST_COLDATA_VAL
//   and SCALAR_VARIANT_VAL : form 3. Variants: 0 the scalar form alone; 1 also the column form on a second copy of
//   the data (output tile 1); 2 the column-broadcast op runs between the load and the scalar multiply (output tile 1
//   is the column form); 3 exp_tile_init runs after the load, then the init and the load repeat.
//   NUM_OUT_TILES_VAL : 1 or 2 output tiles.
//
// DST layout: forms 0..2: dst[0] = data tile (in place, the output), dst[1] = the broadcast tile. Form 3:
// dst[DST_DATA_VAL] = data (in place, output tile 0), dst[DST_SCALAR_VAL] = the scalar tile (element [0, 0]),
// dst[DST_COLDATA_VAL] = a second copy of the data for the column form (output tile 1).

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
#ifndef SCALAR_ITER_VAL
#define SCALAR_ITER_VAL 8
#endif
#ifndef SCALAR_MODE_VAL
#define SCALAR_MODE_VAL 0
#endif
#ifndef DST_DATA_VAL
#define DST_DATA_VAL 0
#endif
#ifndef DST_SCALAR_VAL
#define DST_SCALAR_VAL 1
#endif
#ifndef DST_COLDATA_VAL
#define DST_COLDATA_VAL 2
#endif
#ifndef SCALAR_VARIANT_VAL
#define SCALAR_VARIANT_VAL 0
#endif
#ifndef NUM_OUT_TILES_VAL
#define NUM_OUT_TILES_VAL 1
#endif

namespace {

constexpr uint32_t kForm = FORM_VAL;
constexpr uint32_t kNumOutTiles = NUM_OUT_TILES_VAL;

constexpr uint32_t kDstData = (kForm == 3) ? DST_DATA_VAL : 0;
constexpr uint32_t kDstBcast = (kForm == 3) ? DST_SCALAR_VAL : 1;
constexpr uint32_t kDstColData = DST_COLDATA_VAL;

constexpr ckernel::BroadcastType kBcastDim =
    (BCAST_DIM_VAL == 0) ? ckernel::BroadcastType::COL : ckernel::BroadcastType::ROW;

constexpr ckernel::EltwiseBinaryType kBcastOp = (BINOP_VAL == 0)   ? ckernel::EltwiseBinaryType::ELWADD
                                                : (BINOP_VAL == 1) ? ckernel::EltwiseBinaryType::ELWSUB
                                                                   : ckernel::EltwiseBinaryType::ELWMUL;

constexpr ckernel::VectorMode kScalarMode = (SCALAR_MODE_VAL == 0)   ? ckernel::VectorMode::RC
                                            : (SCALAR_MODE_VAL == 1) ? ckernel::VectorMode::R
                                                                     : ckernel::VectorMode::C;
constexpr int kScalarIterations = SCALAR_ITER_VAL;
constexpr uint32_t kScalarVariant = SCALAR_VARIANT_VAL;

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
    if constexpr (kForm == 3) {
        sfpu_bcast_scalar_init();
    } else {
        sfpu_bcast_init<kBcastDim>();
    }

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
    } else {
        if constexpr (kNumOutTiles == 2) {
            copy_tile(input_a_cb_id, 0, kDstColData);
        }
        if constexpr (kScalarVariant == 0) {
            sfpu_bcast_scalar_load(kDstBcast);
            sfpu_mul_bcast_scalar<kScalarIterations>(kDstData, kScalarMode);
        } else if constexpr (kScalarVariant == 1) {
            sfpu_bcast_scalar_load(kDstBcast);
            sfpu_mul_bcast_scalar<kScalarIterations>(kDstData, kScalarMode);
            // The load runs sfpi code, which may take the column broadcast's mask register: its init again first.
            sfpu_bcast_col_init();
            sfpu_mul_bcast_col(kDstColData, kDstBcast);
        } else if constexpr (kScalarVariant == 2) {
            // The column-broadcast init and op between the load and the use: the parked value survives both.
            sfpu_bcast_scalar_load(kDstBcast);
            sfpu_bcast_col_init();
            sfpu_mul_bcast_col(kDstColData, kDstBcast);
            sfpu_mul_bcast_scalar<kScalarIterations>(kDstData, kScalarMode);
        } else {
            // exp_tile_init programs the constant registers the parked value lives in: init and load again.
            sfpu_bcast_scalar_load(kDstBcast);
            exp_tile_init<false>();
            exp_tile<false>(kDstColData);
            sfpu_bcast_scalar_init();
            sfpu_bcast_scalar_load(kDstBcast);
            sfpu_mul_bcast_scalar<kScalarIterations>(kDstData, kScalarMode);
        }
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(kDstData, out_cb_id, 0);
    if constexpr (kNumOutTiles == 2) {
        pack_tile(kDstColData, out_cb_id, 1);
    }

    tile_regs_release();

    input_a_cb.pop_front(NUM_IN_TILES);
    input_b_cb.pop_front(NUM_IN_TILES);
    out_cb.push_back(kNumOutTiles);
}
