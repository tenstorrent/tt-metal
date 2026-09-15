// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// copy_dest_values with the DESTINATION DST slot already occupied.
//
// The primitive stages one DST tile into another so a later op can transform
// the copy while the original stays packable. The case worth testing is
// therefore the one where the destination slot is live with another op's data
// and the copy has to overwrite it -- not an isolated round trip.
//
// Both DST[0] and DST[1] are written before the copy, from different inputs, so
// they hold distinguishable values. Then DST[0] is copied over DST[1] and BOTH
// slots are packed out. The contract is simply that the two packed tiles are
// then bit-identical.
//
// PRECEDING_MODE selects what wrote the slots, which is the variable this is
// really about -- whether the defect depends on the KIND of preceding op rather
// than on mere occupancy:
//   0  datacopy writes DST[0] and DST[1]
//   1  matmul writes DST[0] and DST[1] (FPU accumulate-into-dest path)
//   2  datacopy writes DST[1], then a matmul writes DST[0] -- so the value
//      sitting in the destination slot came from an op TWO steps back, and the
//      immediately preceding op is a matmul that also reconfigured unpack/math
//
// SKIP_COPY omits the copy entirely. That run packs the un-overwritten
// destination slot, which gives the host the "before" value without needing a
// host-side model of what a matmul produces.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

#if PRECEDING_MODE == 0
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
#else
    // The matmul modes need the reversed src order, as tests/.../compute/matmul.cpp does.
    compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
#endif

    cb0.wait_front(1);
    cb1.wait_front(1);

    tile_regs_acquire();

#if PRECEDING_MODE == 0
    copy_init(tt::CBIndex::c_0);
    copy_tile(tt::CBIndex::c_0, 0, 0);
    copy_init(tt::CBIndex::c_1);
    copy_tile(tt::CBIndex::c_1, 0, 1);
#elif PRECEDING_MODE == 1
    // A*B and B*A differ, so the two slots stay distinguishable.
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
    matmul_init(tt::CBIndex::c_1, tt::CBIndex::c_0);
    matmul_tiles(tt::CBIndex::c_1, tt::CBIndex::c_0, 0, 0, 1);
#else
    copy_init(tt::CBIndex::c_1);
    copy_tile(tt::CBIndex::c_1, 0, 1);
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#endif

#ifndef SKIP_COPY
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float16_b>(0 /*idst_in*/, 1 /*idst_out*/);
#endif

    tile_regs_commit();
    tile_regs_wait();

    // Tile 0 of the output is DST[0] (the copy source), tile 1 is DST[1] (the
    // copy destination). The host compares them to each other.
    cb16.reserve_back(2);
    pack_tile(0, tt::CBIndex::c_16);
    pack_tile(1, tt::CBIndex::c_16);
    cb16.push_back(2);

    tile_regs_release();

    cb0.pop_front(1);
    cb1.pop_front(1);
}
