// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "kernel_thread_globals.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#endif

#include "api/debug/dprint.h"

void kernel_main() {
    uintptr_t src0_addr = get_arg_val<uint32_t>(0);
    uintptr_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t bcast_B = get_arg_val<uint32_t>(8);
    uint32_t reader_id = 0;
    uint32_t num_readers = 1;
#ifdef ARCH_QUASAR
    uint32_t producer_mask = get_arg_val<uint32_t>(9);
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    reader_id = static_cast<uint32_t>(__builtin_popcount(producer_mask & ((1u << hartid) - 1u)));
    num_readers = static_cast<uint32_t>(__builtin_popcount(producer_mask));
#endif

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
        for (uint32_t mt = 0; mt < Mt; mt ++) { // row of in0
            uint32_t itileB = itileB_batch;
            for (uint32_t nt = 0; nt < Nt; nt++) { // col of in1
                for (uint32_t kt = 0; kt < Kt; kt++) { // col of in0, row of in1
                    // Read A's tile at (mt, kt)
                    {
#ifdef ARCH_QUASAR
                        if (mt % num_readers == reader_id) {
                            DPRINT << "reader " << reader_id
                                << " mt " << mt << " nt " << nt << " kt " << kt
                                << " itileA " << itileA
                                << " reserve_back: dfb0 to " << dfb0.get_write_ptr() << ENDL();
                            dfb0.reserve_back(onetile);
                            uint32_t l1_write_addr_in0 = dfb0.get_write_ptr();
                            noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                            noc.async_read_barrier();
                            DPRINT << "reader " << reader_id << " push_back: dfb0" << ENDL();
                            dfb0.push_back(onetile);
                        }
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
                        if (mt % num_readers == reader_id && kt % num_readers == reader_id) {
                            DPRINT << "reader " << reader_id
                                    << " mt " << mt << " nt " << nt << " kt " << kt
                                    << " itileB " << itileB
                                << " reserve_back: dfb1 to " << dfb1.get_write_ptr() << ENDL();
                            dfb1.reserve_back(onetile);
                            uint32_t l1_write_addr_in1 = dfb1.get_write_ptr();
                            noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                            noc.async_read_barrier();
                            DPRINT << "reader " << reader_id << " push_back: dfb1" << ENDL();
                            dfb1.push_back(onetile);
                        }
#else
                        cb_reserve_back(cb_id_in1, onetile);
                        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                        noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in1, onetile);
#endif
                    }

                    itileA += 1;   // A is MK
                    itileB += Nt;  // B is KN, so to get k++ we stride by Nt
                }  // Kt loop
                itileB -= KtNt;
                itileB += 1;
                itileA -= Kt;
            }  // Nt loop
            itileA += Kt;  // A is MK, advance by num_readers rows
        }  // Mt loop
        itileA_batch += MtKt;
        if (bcast_B == 0) {
            itileB_batch += KtNt;
        }
    }  // batch loop
    DPRINT << "reader " << reader_id << " finish dfb0 and dfb1" << ENDL();
    dfb0.finish();
    DPRINT << "reader " << reader_id << " finish dfb0" << ENDL();
    dfb1.finish();
    DPRINT << "reader " << reader_id << " finish dfb0 and dfb1 done" << ENDL();
}
