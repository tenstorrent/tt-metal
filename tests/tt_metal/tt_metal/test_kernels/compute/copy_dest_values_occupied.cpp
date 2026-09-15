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
//   3  like 2, but the op writing DST[0] is a ROW-broadcast eltwise add, and
//      the copy is the RAW SFPU_BINARY_CALL with ITERATIONS=2 and
//      VectorMode::R rather than the public wrapper's 8 / RC. Under R/2 most
//      of the destination tile is out of scope BY DESIGN and legitimately
//      keeps its prior contents, so the host measures the copied footprint
//      instead of demanding whole-tile equality.
//   4  mode 3's preceding op (ROW-broadcast add) but the public wrapper's
//      whole-tile copy, to separate "which op wrote DST[0]" from "which form
//      of the copy call".
//
// FOLLOWING_SFPU decides what happens to DST[1] AFTER the copy, and it is the
// axis that matters most. With 0 the next thing to touch DST[1] is the PACK
// thread, which arrives through the MATH_PACK semaphore handshake. With 1 or 2
// the next thing is another SFPU op reading DST[1] directly -- and nothing
// stalls SFPU on SFPU: _llk_math_eltwise_sfpu_start_ issues
// TTI_STALLWAIT(STALL_SFPU, MATH), which waits on the FPU, not on the vector
// unit. So an SFPU store into DST[1] followed immediately by an SFPU load of
// DST[1] has no barrier between them.
//   0  nothing; pack straight away
//   1  negative_tile(1)
//   2  sigmoid_tile(1)
//
// QUEUE_DEPTH deepens the Tensix instruction stream immediately before the
// copy, with SFPU ops on OTHER DST slots. Each of those is itself a
// _llk_math_eltwise_unary_sfpu_params_, so each reprograms
// DEST_TARGET_REG_CFG_MATH_Offset -- the register the copy's destination
// address comes from. That register is written by a RISC-V MMIO TT_SETC16 and
// consumed by Tensix SFPLOAD/SFPSTORE, and _llk_math_eltwise_sfpu_start_'s
// TTI_STALLWAIT(STALL_SFPU, MATH) waits on the FPU, not on the config write.
// So if that write can be consumed late, the copy stores at the PREVIOUS op's
// base and the real DST[1] keeps its prior contents. Slots 2 and 3 are used so
// the last value left in the register is base + 2*64 or base + 3*64, which
// would put a mis-addressed store somewhere the host can notice.
//
// PRIOR_SECTION runs a whole extra tile_regs_acquire/commit/wait/release cycle
// BEFORE the one under test, with an SFPU op inside it. Under half-sync DST that
// puts the op under test in the other DST half, and leaves the previous half's
// base in DEST_TARGET_REG_CFG_MATH_Offset. Every other variant here has exactly
// one DST section, so a store that used the previous section's base would have
// nowhere wrong to go; with two sections it does. This is the shape of the call
// site being modelled, where the op that filled the destination slot ran in an
// earlier section entirely.
//
// SKIP_COPY omits the copy entirely. That run packs the un-overwritten
// destination slot, which gives the host the "before" value without needing a
// host-side model of what a matmul produces.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
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

    // A leading tile is always packed so the host sees the same layout either
    // way; with PRIOR_SECTION it comes from a genuinely separate DST section.
    tile_regs_acquire();
    copy_init(tt::CBIndex::c_1);
    copy_tile(tt::CBIndex::c_1, 0, 2);
#if PRIOR_SECTION
    negative_tile_init();
    negative_tile(2);
#endif
    tile_regs_commit();
    tile_regs_wait();
    cb16.reserve_back(1);
    pack_tile(2, tt::CBIndex::c_16);
    cb16.push_back(1);
    tile_regs_release();

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
#elif PRECEDING_MODE == 2
    copy_init(tt::CBIndex::c_1);
    copy_tile(tt::CBIndex::c_1, 0, 1);
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#else
    // Modes 3 and 4: the stale value lands first, then a ROW-broadcast add
    // writes DST[0] -- the same shape as the call site this is modelled on.
    copy_init(tt::CBIndex::c_1);
    copy_tile(tt::CBIndex::c_1, 0, 1);
    add_bcast_rows_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    add_tiles_bcast<BroadcastType::ROW>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
#endif

#if QUEUE_DEPTH > 0
    negative_tile_init();
    for (uint32_t q = 0; q < QUEUE_DEPTH; ++q) {
        negative_tile(2);
        negative_tile(3);
    }
#endif

#ifndef SKIP_COPY
#if PRECEDING_MODE == 3
    // The raw call at VectorMode::R with a settable ITERATIONS, plus a switch
    // between the two functor overloads. ITERATIONS is what `dst_reg++` is
    // supposed to turn into row advance, so the copied footprint should scale
    // with it; SFPI_OVERLOAD picks the sfpi dst_reg[] body instead of the
    // TT_SFPLOAD/TT_SFPSTORE one, and the two must agree.
    copy_dest_values_init();
#if SFPI_OVERLOAD
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (false /*APPROXIMATE*/, RAW_ITERATIONS),
        0,
        1,
        0,
        VectorMode::R)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (DataFormat::Float16_b, false, RAW_ITERATIONS),
        0,
        1,
        0,
        VectorMode::R)));
#endif
#else
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float16_b>(0 /*idst_in*/, 1 /*idst_out*/);
#endif
#endif

#if FOLLOWING_SFPU == 1
    negative_tile_init();
    negative_tile(1);
#elif FOLLOWING_SFPU == 2
    sigmoid_tile_init();
    sigmoid_tile(1);
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
