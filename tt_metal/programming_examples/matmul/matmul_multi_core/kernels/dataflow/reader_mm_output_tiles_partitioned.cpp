// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    // Arguments are read by the names declared in the host code.
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);
    uint32_t output_tile_start_id = get_arg(args::output_tile_start_id);  // starting tile ID for output tiles
    uint32_t num_output_tiles = get_arg(args::num_output_tiles);          // number of output tiles to read

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Declare address in which we stored the source matrices. We have set the exact same format between DFBs and DRAM
    // tensors in the host code, so we can use the same address for both DRAM and DFBs.
    // The source tensors are bound to this kernel by name in the host code.

    const auto a = TensorAccessor(tensor::a);

    const auto b = TensorAccessor(tensor::b);

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
                dfb_in0.reserve_back(1);
                uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();
                noc_async_read_page(tile_A, a, l1_write_addr_in0);
                noc_async_read_barrier();
                dfb_in0.push_back(1);
            }

            // Read B's tile at (k, out_col)
            uint32_t tile_B = k * Nt + out_col;  // B is KN, so we stride by Nt
            {
                dfb_in1.reserve_back(1);
                uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
                noc_async_read_page(tile_B, b, l1_write_addr_in1);
                noc_async_read_barrier();
                dfb_in1.push_back(1);
            }
        }
    }
}
