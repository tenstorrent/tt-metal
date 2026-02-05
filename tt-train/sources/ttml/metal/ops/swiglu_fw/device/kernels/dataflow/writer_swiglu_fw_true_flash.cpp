// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TRUE FLASH SwiGLU WRITER KERNEL
//
// This writer is adapted for the True Flash algorithm where Y is produced
// as a complete row (all Wt tiles) after all k_blocks are processed.
//
// Original writer: writes Y in c_blocks as they complete
// True Flash writer: writes complete Y row after all k_block iterations
// ============================================================================

#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, :]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);  // Full output width in tiles

void kernel_main() {
    uint32_t ra = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(ra++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_y_idx);

    // Address generator
    constexpr auto y_args = TensorAccessorArgs<2>();
    const auto y_address_generator = TensorAccessor(y_args, y_addr, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;

    // ================== TRUE FLASH WRITER ==================
    // In True Flash, Y[r, :] is produced as a complete row after all k_blocks.
    // We write the entire row at once instead of writing c_blocks separately.
    //
    // Loop structure:
    //   for r in rows:
    //     wait for complete Y[r, :] (Wt tiles)
    //     write Y[r, :] to DRAM
    // ============================================================================
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Wait for complete Y row (all Wt tiles)
        cb_wait_front(cb_y_idx, Wt);

        // Write entire Y row to DRAM
        const uint32_t start_tile_idx = r * Wt;

        // Get L1 write pointer for the CB
        uint32_t l1_read_addr = get_read_ptr(cb_y_idx);

        // Write tiles sequentially (could be optimized with async writes)
        for (uint32_t c = 0; c < Wt; ++c) {
            const uint32_t tile_idx = start_tile_idx + c;
            uint64_t dst_noc_addr = y_address_generator.get_noc_addr(tile_idx);
            noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
            l1_read_addr += tile_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_y_idx, Wt);
    }
}
