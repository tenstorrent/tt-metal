// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"  // unary_bcast; 32-bit CBs route it onto the unpack-to-dest branch
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Minimal reproducer for the stale-dest-offset bug in the 32-bit unpack-to-dest datacopy branch
// (_llk_math_eltwise_unary_datacopy_, unpack_to_dest && is_32bit_input).
//
// BCAST_DIM_VAL selects the operation under test: 0 = ROW, 1 = COL, 2 = SCALAR broadcast (all three
// sequences in that branch address Dst the same way), 3 = NONE, i.e. a plain 32-bit unpack-to-dest
// copy_tile. Only the broadcast modes reproduce the bug; NONE is a smoke check that the plain copy
// still lands correctly with a dirty offset, and passes with or without the fix on both
// architectures. Blackhole's budabackend/#2730 ZEROACC zero-flag-clear loop does run in this branch
// for every mode including plain copies, but it takes an absolute block index, so the stale offset
// can only reach it through the bank half-select -- out of range for any tile-granular offset.
//
// Sequence, repeated NUM_ITERS_VAL times (each iteration is its own acquire, so the run alternates
// dest banks and, from the second visit of a bank onward, lands on rows the packer has already
// drained and zero-flagged -- the state a displaced Blackhole flag-clear corrupts):
//
//   (1) copy_tile(c_0 [Float16_b], 0, /*idst=*/1)
//       Any SrcRegs-path datacopy: its set_dst_write_addr<..., UnpackDestination::SrcRegs>
//       programs DEST_TARGET_REG_CFG_MATH_Offset to dest_bank_base + 1*64 and LEAVES it there.
//
//   (2) unary_bcast<DIM>(c_1 [Float32, UnpackToDestFp32], 0, /*idst=*/0)   [or copy_tile for NONE]
//       The 32-bit unpack-to-dest branch. Its broadcast MOVD2B/MOVB2D carry bank-local immediates
//       and the hardware ADDS the math dest offset to every one of them -- but this branch never
//       reprograms that offset (its set_dst_write_addr<..., UnpackDestination::DestReg> only
//       mailboxes the write address to the UNPACKER).
//
// Expected:  DST[0] = the c_1 tile broadcast along DIM (or copied verbatim for NONE);
//            DST[1] = the copied c_0 tile, untouched.
// Observed (broadcast modes, without the fix): the broadcast is displaced by the stale +64 -- it
//            reads its source rows from DST[1] and broadcasts them back over DST[1], while DST[0]
//            keeps the raw, unbroadcast c_1 tile the unpacker deposited.
//
// Both DST slots are packed out per iteration (c_16, Float32) so the host sees any displacement.

namespace {
constexpr bool kPlainCopy = (BCAST_DIM_VAL == 3);
constexpr BroadcastType kBcastDim = (BCAST_DIM_VAL == 0)   ? BroadcastType::ROW
                                    : (BCAST_DIM_VAL == 1) ? BroadcastType::COL
                                                           : BroadcastType::SCALAR;
}  // namespace

void kernel_main() {
    constexpr auto cb_copy = tt::CBIndex::c_0;   // Float16_b data tile, copied to DST slot 1
    constexpr auto cb_bcast = tt::CBIndex::c_1;  // Float32 tile (UnpackToDestFp32), lands in DST slot 0
    constexpr auto cb_out = tt::CBIndex::c_16;   // Float32, two tiles per iteration: DST[0] then DST[1]

    CircularBuffer copy_cb(cb_copy);
    CircularBuffer bcast_cb(cb_bcast);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_copy, cb_out);

    for (uint32_t iter = 0; iter < NUM_ITERS_VAL; ++iter) {
        copy_cb.wait_front(1);
        bcast_cb.wait_front(1);
        out_cb.reserve_back(2);

        tile_regs_acquire();

        // (1) Dirty the math dest offset with a copy to a NONZERO dst index.
        copy_init(cb_copy);
        copy_tile(cb_copy, 0, /*idst=*/1);

        // (2) 32-bit unpack-to-dest op into dst index 0. The srcA reconfig retargets the unpacker
        // between the two CBs' formats (Float16_b -> Float32) and back, exactly as a real kernel
        // interleaving the two ops would.
        reconfig_data_format_srca(cb_copy, cb_bcast);
        if constexpr (kPlainCopy) {
            copy_init(cb_bcast);
            copy_tile(cb_bcast, 0, /*idst=*/0);
        } else {
            unary_bcast_init<kBcastDim>(cb_bcast);
            unary_bcast<kBcastDim>(cb_bcast, 0, /*idst=*/0);
            unary_bcast_uninit<kBcastDim>(cb_bcast);
        }
        reconfig_data_format_srca(cb_bcast, cb_copy);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);  // expected: the broadcast result (or the copied c_1 tile for NONE)
        pack_tile(1, cb_out);  // expected: the copied c_0 tile, untouched
        tile_regs_release();

        copy_cb.pop_front(1);
        bcast_cb.pop_front(1);
        out_cb.push_back(2);
    }
}
