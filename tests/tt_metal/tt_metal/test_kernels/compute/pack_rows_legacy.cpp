// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) pack_rows baseline: per tile, copy c_0 -> DST[0] then row-pack the full tile (num_rows = 64,
// a whole 32x32 tile: 64 rows x 16 datums) from DST to c_16 in row-major order via the legacy CB-id
// pack_rows(idst, ocb, output_index). Regression baseline for the id-free variant pack_rows_2_0.cpp, which
// differs ONLY in the row-pack call (experimental::pack_rows). copy_tile / compute_kernel_hw_startup /
// pack_rows_init / pack_rows_uninit stay legacy in BOTH kernels so the differential isolates the row pack.
// num_rows is fixed at 64 (full tile) so the entire output tile is written -- no uninitialized bytes that
// could differ between the two runs. Compile-time arg 0 = total tile count (per-tile 1->1). Output must be
// bit-for-bit identical to the 2.0 kernel.
void kernel_main() {
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr std::uint32_t num_rows = 64;  // full tile (TILE_R*TILE_C / FACE_C = 32*32/16)

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

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
        pack_rows(0, tt::CBIndex::c_16, 0);
        tile_regs_release();

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    pack_rows_uninit();
}
