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

// Minimal reproducer for the stale-dest-offset bug in the 32-bit unpack-to-dest broadcast
// (_llk_math_eltwise_unary_datacopy_, unpack_to_dest && is_32bit_input branch).
//
// Sequence, all inside ONE tile_regs_acquire window:
//
//   (1) copy_tile(c_0 [Float16_b], 0, /*idst=*/1)
//       Any SrcRegs-path datacopy: its set_dst_write_addr<..., UnpackDestination::SrcRegs>
//       programs DEST_TARGET_REG_CFG_MATH_Offset to dest_bank_base + 1*64 and LEAVES it there.
//
//   (2) unary_bcast<ROW>(c_1 [Float32, UnpackToDestFp32], 0, /*idst=*/0)
//       The 32-bit unpack-to-dest broadcast. Its MOVD2B/MOVB2D sequence addresses Dst with
//       immediates (dst_index*64 + row), and the hardware ADDS the math dest offset to every
//       one of them -- but this branch never reprograms that offset (its
//       set_dst_write_addr<..., UnpackDestination::DestReg> only mailboxes the write address
//       to the UNPACKER).
//
// Expected:  DST[0] = row 0 of the c_1 tile replicated down all 32 rows; DST[1] = the copied
//            c_0 tile, untouched.
// Observed:  the broadcast is displaced by the stale +64 -- it reads its source rows from
//            DST[1] (the copied c_0 tile) and broadcasts them back over DST[1], while DST[0]
//            is left holding the raw, unbroadcast c_1 tile the unpacker deposited.
//
// Both DST slots are packed out (c_16, Float32) so the host sees the displacement directly.

void kernel_main() {
    constexpr auto cb_copy = tt::CBIndex::c_0;   // Float16_b data tile, copied to DST slot 1
    constexpr auto cb_bcast = tt::CBIndex::c_1;  // Float32 bcast tile (UnpackToDestFp32), row 0 meaningful
    constexpr auto cb_out = tt::CBIndex::c_16;   // Float32, two tiles: DST[0] then DST[1]

    CircularBuffer copy_cb(cb_copy);
    CircularBuffer bcast_cb(cb_bcast);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_copy, cb_out);

    copy_cb.wait_front(1);
    bcast_cb.wait_front(1);
    out_cb.reserve_back(2);

    tile_regs_acquire();

    // (1) Dirty the math dest offset with a copy to a NONZERO dst index.
    copy_init(cb_copy);
    copy_tile(cb_copy, 0, /*idst=*/1);

    // (2) 32-bit unpack-to-dest ROW broadcast into dst index 0. The srcA reconfig retargets the
    // unpacker between the two CBs' formats (Float16_b -> Float32) and back, exactly as a real
    // kernel interleaving the two ops would.
    reconfig_data_format_srca(cb_copy, cb_bcast);
    unary_bcast_init<BroadcastType::ROW>(cb_bcast);
    unary_bcast<BroadcastType::ROW>(cb_bcast, 0, /*idst=*/0);
    unary_bcast_uninit<BroadcastType::ROW>(cb_bcast);
    reconfig_data_format_srca(cb_bcast, cb_copy);

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, cb_out);  // expected: the broadcast result
    pack_tile(1, cb_out);  // expected: the copied c_0 tile, untouched
    tile_regs_release();

    copy_cb.pop_front(1);
    bcast_cb.pop_front(1);
    out_cb.push_back(2);
}
