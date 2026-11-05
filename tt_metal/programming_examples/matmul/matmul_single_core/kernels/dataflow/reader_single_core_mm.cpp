// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    // Arguments are read by the names declared in the host code.
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Declare address in which we stored the source matrices. We have set the exact same format between DFBs and DRAM
    // tensors in the host code, so we can use the same address for both DRAM and DFBs.
    // The source tensors are bound to this kernel by name in the host code.
    const auto s0 = TensorAccessor(tensor::in0);
    const auto s1 = TensorAccessor(tensor::in1);

    // Loop through the dimensions of the matrices. Read them and push to the dataflow buffers.
    // Dimension names are called M, N and K. `t` in `mt` means tile.
    for (uint32_t mt = 0; mt < Mt; mt++) {
        uint32_t itileB = 0;
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                {                                          // Read A's tile at (mt, kt)
                    uint32_t a_tile_index = mt * Kt + kt;  // A is MK, so we stride by Kt
                    dfb_in0.reserve_back(1);
                    uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();
                    noc_async_read_page(a_tile_index, s0, l1_write_addr_in0);
                    noc_async_read_barrier();
                    dfb_in0.push_back(1);
                }

                {                                          // Read B's tile at (kt, nt)
                    uint32_t b_tile_index = kt * Nt + nt;  // B is KN, so we stride by Nt
                    dfb_in1.reserve_back(1);
                    uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
                    noc_async_read_page(b_tile_index, s1, l1_write_addr_in1);
                    noc_async_read_barrier();
                    dfb_in1.push_back(1);
                }
            }  // Kt loop
        }  // Nt loop
    }  // Mt loop
}
