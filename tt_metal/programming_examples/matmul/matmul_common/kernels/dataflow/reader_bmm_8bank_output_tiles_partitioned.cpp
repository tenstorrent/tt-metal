// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t bcast_B = get_arg_val<uint32_t>(8);  // if 1 we broadcast B to batch
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(9);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(10);
    uint32_t MtNt = get_arg_val<uint32_t>(11);

    // DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    // DPRINT << "src0=" << src0_addr << " src1=" << src1_addr << ENDL();
    // DPRINT << "batch=" << batch << ENDL();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    uint32_t itileA = output_tile_start_id / Nt * Kt;  // input0 row = output row * input0 width

    // Keep track of end of output row and end of output batch
    uint32_t outbatch = output_tile_start_id % MtNt;
    uint32_t itileB_batch = output_tile_start_id % Nt;
    uint32_t itileB = itileB_batch;  // input1 col = output col if we are bcasting
    if (bcast_B == 0) {
        itileB += output_tile_start_id / MtNt * KtNt;  // offset into correct batch if not bcasting
    }

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr, in0_tile_bytes);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, in1_tile_bytes);

    for (uint32_t n = 0; n < num_output_tiles; n++) {
        for (uint32_t kt = 0; kt < Kt; kt++) {
            {  // Read A's tile at (mt, kt)
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }

            {  // Read B's tile at (kt, nt)
                cb_reserve_back(cb_id_in1, onetile);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);
            }
            // DPRINT << "Pushed itileA=" << itileA << " itileB=" << itileB << ENDL();

            itileA += 1;   // A is MK
            itileB += Nt;  // B is KN, so to get k++ we stride by Nt
        }  // Kt loop
        outbatch += 1;
        itileB_batch += 1;
        itileB -= KtNt;  // revert B to previous state before the K loop (to avoid multiplies)
        itileB += 1;     // Move to next B col

        if (itileB_batch == Nt) {
            itileB_batch = 0;
            itileB -= Nt;  // Go back to first column in batch
            if (outbatch == MtNt) {
                if (bcast_B == 0) {
                    itileB += KtNt;  // Move B to start of next batch
                }
                outbatch = 0;
            }
        } else {
            itileA -= Kt;  // resets tileA to kt=0, keep the same mt
        }
    }  // batch loop
}
