// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PolyNorm3 backward — writer kernel
//
// Writes two output tensors to DRAM per row:
//   1. dL/dx — full-width gradient tiles (Wt tiles per row)
//   2. packed_partials — 4 tiles per row containing [dw0, dw1, dw2, db]
//      (later reduced on the host across all rows/cores to produce final dL/dw and dL/db)
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg_idx = 0U;
    const uint32_t dL_dx_output_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packed_partials_output_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto cb_dL_dx_output = tt::CBIndex::c_21;
    constexpr auto cb_packed_partials_output = tt::CBIndex::c_22;
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t dL_dx_Wt = get_compile_time_arg_val(1);
    constexpr uint32_t packed_partials_wt = get_compile_time_arg_val(2);

    const uint32_t dL_dx_tile_bytes = get_tile_size(cb_dL_dx_output);
    const uint32_t packed_partials_tile_bytes = get_tile_size(cb_packed_partials_output);

    constexpr auto dL_dx_output_args = TensorAccessorArgs<3>();
    const auto dL_dx_output_addr_generator = TensorAccessor(dL_dx_output_args, dL_dx_output_address, dL_dx_tile_bytes);

    constexpr auto packed_partials_output_args =
        TensorAccessorArgs<dL_dx_output_args.next_compile_time_args_offset()>();
    const auto packed_partials_output_addr_generator =
        TensorAccessor(packed_partials_output_args, packed_partials_output_address, packed_partials_tile_bytes);

    for (uint32_t r = start_row; r < (start_row + num_rows_to_process); ++r) {
        // Write dL/dx gradient tiles for this row
        write_full_row_tiles(
            cb_dL_dx_output, dL_dx_output_addr_generator, dL_dx_Wt, block_size, dL_dx_tile_bytes, r * dL_dx_Wt);

        // Write packed partial-gradient tiles [dw0, dw1, dw2, db]
        write_full_row_tiles(
            cb_packed_partials_output,
            packed_partials_output_addr_generator,
            packed_partials_wt,
            block_size,
            packed_partials_tile_bytes,
            r * packed_partials_wt);
    }
}
