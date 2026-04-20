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
//   BCAST_DIM_VAL : 0 = BCAST_COL, 1 = BCAST_ROW
//   BINOP_VAL     : 0 = ADD,       1 = SUB,      2 = MUL
//
// DST layout used by this kernel:
//   dst[0] = data tile   (input A, also output written in-place)
//   dst[1] = broadcast tile (input B; col 0 read for COL, row 0 read for ROW)

namespace {

constexpr uint32_t kDstData = 0;
constexpr uint32_t kDstBcast = 1;

ALWI void sfpu_bcast_init_selected() {
#if BCAST_DIM_VAL == 0
    sfpu_bcast_col_init();
#else
    sfpu_bcast_row_init();
#endif
}

ALWI void sfpu_bcast_apply_selected() {
#if BCAST_DIM_VAL == 0
#if BINOP_VAL == 0
    sfpu_add_bcast_col(kDstData, kDstBcast);
#elif BINOP_VAL == 1
    sfpu_sub_bcast_col(kDstData, kDstBcast);
#elif BINOP_VAL == 2
    sfpu_mul_bcast_col(kDstData, kDstBcast);
#else
#error "Unknown BINOP_VAL"
#endif
#else  // BCAST_ROW
#if BINOP_VAL == 0
    sfpu_add_bcast_row(kDstData, kDstBcast);
#elif BINOP_VAL == 1
    sfpu_sub_bcast_row(kDstData, kDstBcast);
#elif BINOP_VAL == 2
    sfpu_mul_bcast_row(kDstData, kDstBcast);
#else
#error "Unknown BINOP_VAL"
#endif
#endif
}

}  // namespace

void kernel_main() {
    constexpr auto cb_data = tt::CBIndex::c_0;
    constexpr auto cb_bcast = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    experimental::CircularBuffer cb0(cb_data);
    experimental::CircularBuffer cb1(cb_bcast);
    experimental::CircularBuffer cb16(cb_out);

    init_sfpu(cb_data, cb_out);
    sfpu_bcast_init_selected();

    cb0.wait_front(1);
    cb1.wait_front(1);
    cb16.reserve_back(1);

    acquire_dst();

    copy_tile_to_dst_init_short(cb_data);
    copy_tile(cb_data, 0, kDstData);

    copy_tile_to_dst_init_short(cb_bcast);
    copy_tile(cb_bcast, 0, kDstBcast);

    sfpu_bcast_apply_selected();

    pack_tile(kDstData, cb_out);

    release_dst();

    cb0.pop_front(1);
    cb1.pop_front(1);
    cb16.push_back(1);
}
