// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"  // legacy pack_rows_init / pack_rows_uninit (both CB-id-free)
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) pack_rows kernel. IDENTICAL to pack_rows_legacy.cpp except the row pack uses
// experimental::pack_rows, built from an output LLKOperand (data format + tile geometry as NTTPs, absolute L1
// write address the only runtime state). compute_kernel_hw_startup, the input copy_tile, and pack_rows_init /
// pack_rows_uninit stay the legacy CB-id-free API in BOTH kernels so the differential isolates the row pack.
// pack_rows_init(num_rows) programs the row count/counters (no format, no CB); experimental::pack_rows then
// packs num_rows row-major rows from DST[0] to out.l1_address. num_rows is fixed at 64 (full tile) so the whole
// output tile is written. Compile-time arg 0 = total tile count (per-tile 1->1). Output must be bit-for-bit
// identical to the legacy kernel.
void kernel_main() {
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr std::uint32_t num_rows = 64;  // full tile (TILE_R*TILE_C / FACE_C = 32*32/16)

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Compile-time CB accessor -> folded descriptor; the operand bundles descriptor + runtime L1 address.
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);
    pack_rows_init(num_rows);

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        // Row-pack DST[0] to the reserved output tile base (output_index 0 == write_address()).
        experimental::pack_rows(OutOp(out_cb.write_address()), 0 /*idst*/);
        tile_regs_release();

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    pack_rows_uninit();
}
