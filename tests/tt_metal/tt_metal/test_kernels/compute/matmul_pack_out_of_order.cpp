// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

// Guards the out-of-order branch of llk_matmul_pack: DST tile (start_tile_index + i) must land on output
// page (output_tile_index + i), i.e. both the DST start and the page start are honoured per tile.
//
// c_0 tile i is copied to DST[i] (plain datacopy, ntiles even). The two DST halves are then packed swapped,
// each half with one out-of-order llk_matmul_pack call:
//   llk_matmul_pack<.., true>(half, c_16, half, 0)     DST[half .. ntiles-1] -> pages 0 .. half-1
//   llk_matmul_pack<.., true>(0,    c_16, half, half)  DST[0 .. half-1]      -> pages half .. ntiles-1
// so page p must hold DST tile (p + half) % ntiles. A loop that holds the page index constant packs each
// half onto one page; a loop that drops output_tile_index packs DST[i] to page i; a loop that drops the
// start_tile_index rebase packs both halves onto pages half .. ntiles-1. All three differ from the
// expected layout. Not Quasar (no llk_matmul_pack there).
void kernel_main() {
    constexpr uint32_t ntiles = get_compile_time_arg_val(0);
    static_assert(ntiles % 2 == 0, "the two packed halves must be the same size");
    constexpr uint32_t half = ntiles / 2;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    tile_regs_acquire();
    cb0.wait_front(ntiles);
    for (uint32_t i = 0; i < ntiles; ++i) {
        copy_tile(tt::CBIndex::c_0, i, i);
    }
    cb0.pop_front(ntiles);
    tile_regs_commit();

    tile_regs_wait();
    cb16.reserve_back(ntiles);
    PACK((llk_matmul_pack<DST_ACCUM_MODE, true /*out_of_order_output*/, PackMode::Default>(
        half /*start_tile_index*/, tt::CBIndex::c_16, half /*ntiles*/, 0 /*output_tile_index*/)));
    PACK((llk_matmul_pack<DST_ACCUM_MODE, true /*out_of_order_output*/, PackMode::Default>(
        0 /*start_tile_index*/, tt::CBIndex::c_16, half /*ntiles*/, half /*output_tile_index*/)));
    cb16.push_back(ntiles);
    tile_regs_release();
}
