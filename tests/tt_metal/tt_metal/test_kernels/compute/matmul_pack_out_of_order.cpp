// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

// Guards the out-of-order branch of llk_matmul_pack: one call packs `ntiles` consecutive DST tiles to
// `ntiles` consecutive pages of the reserved region of c_16, starting at output_tile_index 0.
//
// c_0 tile i is copied to DST[i] (plain datacopy), then every DST tile is packed by a single
// llk_matmul_pack<DST_ACCUM_MODE, true /*out_of_order_output*/>(0, c_16, ntiles, 0). Out-of-order packing
// addresses each L1 page from the index argument alone (it never advances the CB write-tile pointer), so
// the loop inside llk_matmul_pack has to offset that index per DST tile: with the index held constant every
// DST tile lands on page 0 and the remaining pages are never written.
//
// Expected: c_16 page i == c_0 tile i for every i. Not Quasar (no llk_matmul_pack there).
void kernel_main() {
    constexpr uint32_t ntiles = get_compile_time_arg_val(0);

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
        0 /*start_tile_index*/, tt::CBIndex::c_16, ntiles, 0 /*output_tile_index*/)));
    cb16.push_back(ntiles);
    tile_regs_release();
}
