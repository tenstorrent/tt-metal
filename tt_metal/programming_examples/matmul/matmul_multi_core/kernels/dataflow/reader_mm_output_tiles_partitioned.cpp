// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(5);  // starting tile ID for output tiles
    uint32_t num_output_tiles = get_arg_val<uint32_t>(6);      // number of output tiles to read

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    // Declare address in which we stored the source matrices. We have set the exact same format between CBs and DRAM
    // buffers in the host code, so we can use the same address for both DRAM and CBs.
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr, in0_tile_bytes);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr, in1_tile_bytes);

    // Simple 2D matmul: A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
    for (uint32_t output_tile = 0; output_tile < num_output_tiles; output_tile++) {
        uint32_t current_tile_id = output_tile_start_id + output_tile;

        // Convert linear output tile ID to 2D coordinates
        uint32_t out_row = current_tile_id / Nt;  // Which row in output
        uint32_t out_col = current_tile_id % Nt;  // Which col in output

        // Read all K tiles for this output position
        for (uint32_t k = 0; k < Kt; k++) {
            // Read A's tile at (out_row, k)
            uint32_t tile_A = out_row * Kt + k;  // A is MK, so we stride by Kt
            {
                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(tile_A, a, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, 1);
            }

            // Read B's tile at (k, out_col)
            uint32_t tile_B = k * Nt + out_col;  // B is KN, so we stride by Nt
            {
                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(tile_B, b, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, 1);
            }
        }
    }
}
