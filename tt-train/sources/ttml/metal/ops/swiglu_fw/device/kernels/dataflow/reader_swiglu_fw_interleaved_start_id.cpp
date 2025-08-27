// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <debug/dprint.h>

#include <ostream>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CB indexes must match compute kernel
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);         // output width (C)
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);  // inner dimension (K==P)

// Utility: read contiguous tiles in row-major from DRAM to CB
inline void read_tiles_by_row(
    uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    uint32_t start_idx,
    uint32_t num_tiles,
    const uint32_t tile_bytes) {
    cb_reserve_back(cb_idx, num_tiles);
    uint32_t l1_addr = get_write_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        noc_async_read_tile(start_idx + t, addr_gen, l1_addr);
        l1_addr += tile_bytes;
    }
}

// Utility: read contiguous tiles in col-major from DRAM to CB
inline void read_tiles_by_col(
    uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    uint32_t start_idx,
    uint32_t num_tiles,
    const uint32_t tile_bytes,
    uint32_t stride) {
    cb_reserve_back(cb_idx, num_tiles);
    uint32_t l1_addr = get_write_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t tile_idx = start_idx + t * stride;
        noc_async_read_tile(tile_idx, addr_gen, l1_addr);
        l1_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t ra = 0;
    uint32_t input_address = get_arg_val<uint32_t>(ra++);  // DRAM base for X
    uint32_t w1_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W1
    uint32_t w2_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W2
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_fmt = get_dataformat(cb_input_idx);

    // Address generators
    const InterleavedAddrGenFast<true> x_ag = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_fmt};
    const InterleavedAddrGenFast<true> w1_ag = {
        .bank_base_address = w1_address, .page_size = tile_bytes, .data_format = data_fmt};
    const InterleavedAddrGenFast<true> w2_ag = {
        .bank_base_address = w2_address, .page_size = tile_bytes, .data_format = data_fmt};

    DPRINT << "Start row: " << start_row << ENDL();
    const uint32_t end_row = start_row + num_rows_to_process;
    DPRINT << "End row: " << end_row << ENDL();
    DPRINT << "Block size: " << block_size << ENDL();

    DPRINT << "Wt (C dim): " << Wt << ENDL();
    DPRINT << "hidden_Wt (hidden dim): " << hidden_Wt << ENDL();

    // ================== Loop structure matches compute kernel ==================
    for (uint32_t r = start_row; r < end_row; ++r) {
        DPRINT << "Processing row " << r << ENDL();
        // ---- Outer loop: c_blocks ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            /*
            DPRINT << "  c_block_start: " << c_block_start << ENDL();
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);
            DPRINT << "  c_block_size: " << c_block_size << ENDL();
            */

            // ---- Mid loop: k_blocks ----
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                DPRINT << "    k_block_start: " << k_block_start << ENDL();
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : (hidden_Wt - k_block_start);
                DPRINT << "    k_block_size: " << k_block_size << ENDL();

                // --------------------
                // STEP A: Stream X + W1 for all k_local in this k_block
                // --------------------
                for (uint32_t k_local = 0; k_local < k_block_size; ++k_local) {
                    uint32_t k_global = k_block_start + k_local;
                    DPRINT << "      k_local: " << k_local << " (k_global: " << k_global << ")" << ENDL();

                    // p_blocks along inner dimension P
                    for (uint32_t p_block_start = 0; p_block_start < hidden_Wt; p_block_start += block_size) {
                        const uint32_t p_block_size =
                            (p_block_start + block_size <= hidden_Wt) ? block_size : (hidden_Wt - p_block_start);
                        DPRINT << "        p_block_start: " << p_block_start << ENDL();

                        // --- Read p_block_size tiles of X[r, p_block]
                        // X is stored row-major: offset = row * hidden_Wt + p_block_start
                        DPRINT << "          Reading " << p_block_size << " tiles of X at row " << r << ENDL();
                        uint32_t x_tile_start = r * hidden_Wt + p_block_start;
                        read_tiles_by_row(cb_input_idx, x_ag, x_tile_start, p_block_size, tile_bytes);
                        noc_async_read_barrier();
                        cb_push_back(cb_input_idx, p_block_size);
                        DPRINT << "          Pushed " << p_block_size << " tiles of X to CB" << ENDL();

                        // --- Read p_block_size tiles of W1[p_block, k_global]
                        // W1 is stored col-major: stride = hidden_Wt (P dim)
                        DPRINT << "          Reading " << p_block_size << " tiles of W1 at k " << k_global << ENDL();
                        uint32_t w1_tile_start = p_block_start * hidden_Wt + k_global;
                        read_tiles_by_col(cb_w1_idx, w1_ag, w1_tile_start, p_block_size, tile_bytes, hidden_Wt);
                        noc_async_read_barrier();
                        cb_push_back(cb_w1_idx, p_block_size);
                        DPRINT << "          Pushed " << p_block_size << " tiles of W1 to CB" << ENDL();

                        // Compute kernel will consume cb_input_idx + cb_w1_idx together
                    }
                }
                DPRINT << "    Finished streaming X and W1 for k_block starting at " << k_block_start << ENDL();
                continue;
                /*
                // --------------------
                // STEP B: Stream W2[k_block, c_block]
                // --------------------
                // W2 is stored col-major: each increment in k is stride=size_in_C=Wt
                uint32_t w2_tile_start = k_block_start * Wt + c_block_start;
                read_tiles_by_col(cb_w2_idx, w2_ag, w2_tile_start, c_block_size, tile_bytes, Wt);
                noc_async_read_barrier();
                cb_push_back(cb_w2_idx, c_block_size);

                // Compute kernel now has:
                // - cb_xw1_idx filled from M[r,k]'s
                // - cb_w2_idx filled with W2[k_block, c_block]
                // And will perform Phase B accumulation.
                */
            }
        }
    }
}
