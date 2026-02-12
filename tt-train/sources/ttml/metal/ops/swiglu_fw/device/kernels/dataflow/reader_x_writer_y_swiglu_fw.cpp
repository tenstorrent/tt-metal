// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Dual-NOC X Reader + Y Writer Kernel (RISCV_1 / NOC0)
//
// This kernel runs on ALL cores (sender + receivers) on RISCV_1.
// It handles:
//   - Phase A: Read X tiles from DRAM (for compute to consume)
//   - Phase C: Write Y tiles to DRAM (after compute produces them)
//
// X reads and weight reads/multicast happen CONCURRENTLY on separate
// RISCs and NOCs:
//   RISCV_1 (this kernel, NOC0): X reads + Y writes
//   RISCV_0 (weight kernel, NOC1): weight reads/multicast/receive
//
// This matches the tt-metal matmul 1D mcast architecture where in0
// (activations) are read on RISCV_1 and in1 (weights) on RISCV_0.
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_y_idx = tt::CBIndex::c_10;     // Final Y[r, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t x_address = get_arg_val<uint32_t>(ra++);
    const uint32_t y_address = get_arg_val<uint32_t>(ra++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Address generators for X (read) and Y (write)
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x_address_generator = TensorAccessor(x_args, x_address, tile_bytes);
    const auto y_address_generator = TensorAccessor(y_args, y_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    const uint32_t end_row_for_sync = start_row + max_rows_for_sync;

    // Loop for max_rows_for_sync iterations to stay in sync with multicast.
    // For padding rows (r >= end_row), read last valid X row but don't write Y.
    for (uint32_t r = start_row; r < end_row_for_sync; ++r) {
        const bool is_padding_row = (r >= end_row);

        // ---- Phase A: Read X[r, :] in p_blocks ----
        // For padding rows, read last valid row to keep compute fed
        const uint32_t x_row = is_padding_row ? (end_row - 1) : r;
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;
            const uint32_t x_tile_start = x_row * Wt + p_block_start;
            read_tiles_by_row(cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);
        }

        // ---- Phase B: Nothing (SiLU is compute-only, no dataflow needed) ----

        // ---- Phase C: Write Y[r, :] in c_blocks (actual rows only) ----
        if (!is_padding_row) {
            for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
                const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);
                const uint32_t start_tile_idx = r * Wt + c_block_start;
                write_tiles_by_row(cb_y_idx, y_address_generator, start_tile_idx, c_block_size, tile_bytes, block_size);
            }
        }
    }
}
