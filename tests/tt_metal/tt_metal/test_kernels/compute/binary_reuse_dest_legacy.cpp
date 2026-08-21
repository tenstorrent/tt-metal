// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"  // compute_kernel_hw_startup + CB-id reuse-dest API + tile_regs_*
#include "api/compute/tile_move_copy.h"  // legacy copy_tile (seed DST) + pack_tile
#include "api/dataflow/circular_buffer.h"

// Legacy CB-id eltwise-binary DEST-REUSE ADD kernel. Per tile: seed DST[0] with operand A (c_0) via copy_tile,
// then add_reuse_dest_tiles<DEST_TO_SRCA> folds operand B (c_1) from L1 into DST[0] (DST -> SrcA, c_1 -> SrcB),
// producing A + B, then pack DST[0] -> c_16. This kernel differs from binary_reuse_dest_2_0.cpp ONLY in the
// reuse-dest init + op (CB-id here, id-free LLKOperand there); copy_tile / pack_tile / compute_kernel_hw_startup
// stay the legacy CB-id API in BOTH, so the differential isolates the reuse-dest op.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer c0(tt::CBIndex::c_0);
    CircularBuffer c1(tt::CBIndex::c_1);
    CircularBuffer c16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        c0.wait_front(1);
        c1.wait_front(1);
        c16.reserve_back(1);

        tile_regs_acquire();
        // Seed DST[0] with operand A from c_0 (legacy datacopy; identical in both kernels).
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        // Op under test: DST[0] = DST[0] + c_1  (CB-id reuse-dest API).
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tt::CBIndex::c_1);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tt::CBIndex::c_1, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        c0.pop_front(1);
        c1.pop_front(1);
        c16.push_back(1);
    }
}
