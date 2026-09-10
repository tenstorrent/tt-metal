// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"                   // legacy compute_kernel_hw_startup + tile_regs_*
#include "api/compute/tile_move_copy.h"                   // legacy copy_tile (seed DST) + pack_tile
#include "api/compute/experimental/2_0/eltwise_binary.h"  // id-free reuse-dest API (op under test)
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) eltwise-binary DEST-REUSE ADD kernel. IDENTICAL to binary_reuse_dest_legacy.cpp except the
// reuse-dest init + op use the id-free experimental::add_reuse_dest_{init,tiles} built from an LLKOperand for
// the L1 operand (c_1); the reused operand IS the DST register (runtime index, no LLKOperand). copy_tile /
// pack_tile / compute_kernel_hw_startup stay the legacy CB-id API so the differential isolates the reuse-dest
// op. Per tile: seed DST[0] with A (c_0), then DST[0] = DST[0] + c_1 (DST -> SrcA, c_1 -> SrcB) = A + B, pack
// DST[0] -> c_16. Output must be bit-for-bit identical to the legacy kernel.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer c0(tt::CBIndex::c_0);
    CircularBuffer c1(tt::CBIndex::c_1);
    CircularBuffer c16(tt::CBIndex::c_16);

    // L1 operand B (c_1) as an id-free LLKOperand: format + geometry NTTPs, L1 address the only runtime state.
    constexpr auto in1_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto in1_desc = experimental::to_llk_mem_descriptor(in1_cb);
    using BOp = experimental::LLKOperand<static_cast<DataFormat>(in1_desc.format), in1_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        c0.wait_front(1);
        c1.wait_front(1);
        c16.reserve_back(1);

        tile_regs_acquire();
        // Seed DST[0] with operand A from c_0 (legacy datacopy; identical in both kernels).
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        // Op under test: DST[0] = DST[0] + c_1  (id-free reuse-dest API; c_1 -> BOp, DST reused).
        experimental::add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(BOp(in1_cb.read_address()));
        experimental::add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(BOp(in1_cb.read_address()), 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        c0.pop_front(1);
        c1.pop_front(1);
        c16.push_back(1);
    }
}
