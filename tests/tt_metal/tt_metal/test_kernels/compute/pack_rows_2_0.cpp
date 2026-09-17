// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"  // legacy copy_tile OP (the copy-into-DST op stays CB-id)
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"  // id-free copy_init
#include "api/compute/experimental/2_0/pack.h"            // id-free pack_rows[_init/_uninit]
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) pack_rows kernel. IDENTICAL to pack_rows_legacy.cpp except the compute-init calls + row pack are
// id-free: experimental::copy_init (built from an input LLKOperand) replaces the legacy CB-id copy_tile_init, and
// experimental::pack_rows_init/pack_rows/pack_rows_uninit (built from an output LLKOperand -- data format + tile
// geometry as NTTPs, absolute L1 write address the only runtime state) replace the legacy CB-id-free pack_rows
// calls. compute_kernel_hw_startup and the input copy_tile OP stay the legacy CB-id API. pack_rows_init(num_rows)
// programs the row count/counters (no format, no CB); experimental::pack_rows then packs num_rows row-major rows
// from DST[0] to out.l1_address. num_rows is fixed at 64 (full tile) so the whole output tile is written.
// Compile-time arg 0 = total tile count (per-tile 1->1). Output must be bit-for-bit identical to the legacy kernel.
void kernel_main() {
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr std::uint32_t num_rows = 64;  // full tile (TILE_R*TILE_C / FACE_C = 32*32/16)

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Compile-time CB accessors -> folded descriptors; the operand bundles descriptor + runtime L1 address.
    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    experimental::copy_init(InOp(in_cb.read_address()));
    experimental::pack_rows_init(OutOp(out_cb.write_address()), num_rows);

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

    experimental::pack_rows_uninit(OutOp(out_cb.write_address()));
}
