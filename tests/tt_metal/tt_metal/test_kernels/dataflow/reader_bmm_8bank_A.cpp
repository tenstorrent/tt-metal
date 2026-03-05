// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#endif

#include "api/debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uintptr_t src0_addr = get_arg_val<uint32_t>(0);
    uintptr_t dfb_index = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t bcast_B = get_arg_val<uint32_t>(8);  // if 1 we broadcast B to batch

    // DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    DPRINT << "src0=" << src0_addr << " src1=" << dfb_index << ENDL();
    // DPRINT << "batch=" << batch << ENDL();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::DataflowBuffer dfb0(dfb_index);
    const uint32_t src0_tile_bytes = dfb0.get_entry_size();

    constexpr auto src0_args = TensorAccessorArgs<0>();

    uint32_t itileA_batch = 0;

    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);

    for (uint32_t nb = 0; nb < batch; nb++) {
        uint32_t itileA = itileA_batch;
        for (uint32_t mt = 0; mt < Mt; mt++) {
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Read A's tile at (mt, kt)
                    {
                        dfb0.reserve_back(onetile);
                        uint32_t l1_write_addr_in0 = dfb0.get_write_ptr();
                        noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                        DPRINT << "Read tile " << itileA << " at address " << l1_write_addr_in0 << ENDL();
                        noc.async_read_barrier();
                        dfb0.push_back(onetile);
                    }
                    // DPRINT << "Pushed itileA=" << itileA << " itileB=" << itileB << ENDL();
                    itileA += 1;  // A is MK
                }  // Kt loop
                itileA -= Kt;  // resets tileA to kt=0, keep the same mt
            }  // Nt loop
            itileA += Kt;  // A is MK, advance to next M
        }  // Mt loop
        itileA_batch += MtKt;  // update batch strides
    }  // batch loop
}
