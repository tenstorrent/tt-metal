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
    uintptr_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t bcast_B = get_arg_val<uint32_t>(8);  // if 1 we broadcast B to batch

    // DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    // DPRINT << "src0=" << src0_addr << " src1=" << src1_addr << ENDL();
    // DPRINT << "batch=" << batch << ENDL();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
#ifdef ARCH_QUASAR
    experimental::Noc noc;
    experimental::DataflowBuffer dfb0(0);
    experimental::DataflowBuffer dfb1(1);
    const uint32_t src0_tile_bytes = dfb0.get_entry_size();
    const uint32_t src1_tile_bytes = dfb1.get_entry_size();
#else
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
#endif
    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    uint32_t itileA_batch = 0;
    uint32_t itileB_batch = 0;

    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);

    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);

    for (uint32_t nb = 0; nb < batch; nb++) {
        uint32_t itileA = itileA_batch;
        for (uint32_t mt = 0; mt < Mt; mt++) {
            uint32_t itileB = itileB_batch;
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Read A's tile at (mt, kt)
                    {
#ifdef ARCH_QUASAR
                        dfb0.reserve_back(onetile);
                        uint32_t l1_write_addr_in0 = dfb0.get_write_ptr();
                        noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                        noc.async_read_barrier();
                        dfb0.push_back(onetile);
#else
                        cb_reserve_back(cb_id_in0, onetile);
                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                        noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in0, onetile);
#endif
                    }

                    {  // Read B's tile at (kt, nt)
#ifdef ARCH_QUASAR
                        dfb1.reserve_back(onetile);
                        uint32_t l1_write_addr_in1 = dfb1.get_write_ptr();
                        noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                        noc.async_read_barrier();
                        dfb1.push_back(onetile);
#else
                        cb_reserve_back(cb_id_in1, onetile);
                        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                        noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in1, onetile);
#endif
                    }
                    // DPRINT << "Pushed itileA=" << itileA << " itileB=" << itileB << ENDL();

                    itileA += 1;   // A is MK
                    itileB += Nt;  // B is KN, so to get k++ we stride by Nt
                }  // Kt loop
                itileB -= KtNt;  // revert B to previous state before the K loop (to avoid multiplies)
                itileB += 1;     // B is KN, so here in the end of Nt loop we increment N by 1
                itileA -= Kt;    // resets tileA to kt=0, keep the same mt
            }  // Nt loop
            itileA += Kt;  // A is MK, advance to next M
        }  // Mt loop
        itileA_batch += MtKt;  // update batch strides
        if (bcast_B == 0) {    // don't increment batch if we broadcast matrix B
            itileB_batch += KtNt;
        }
    }  // batch loop
}
