// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);
    uint32_t MtKt = get_arg(args::MtKt);
    uint32_t KtNt = get_arg(args::KtNt);
    uint32_t batch = get_arg(args::batch);
    uint32_t bcast_B = get_arg(args::do_bcast);
    uint32_t batch_start = get_arg(args::batch_start);

    uint32_t reader_id = get_my_thread_id();
    uint32_t num_readers = get_num_threads();

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb0(dfb::src0);
    DataflowBuffer dfb1(dfb::src1);
#ifndef ARCH_QUASAR
    const uint32_t entry_size0 = dfb0.get_entry_size();
    const uint32_t entry_size1 = dfb1.get_entry_size();
#endif

    const auto s0 = TensorAccessor(tensor::src0);
    const auto s1 = TensorAccessor(tensor::src1);

    uint32_t itileA_batch = batch_start * MtKt;
    uint32_t itileB_batch = batch_start * KtNt;

    for (uint32_t nb = 0; nb < batch; nb++) {
        uint32_t itileA = itileA_batch;
        for (uint32_t mt = 0; mt < Mt; mt++) { // row of in0
            uint32_t itileB = itileB_batch;
            for (uint32_t nt = 0; nt < Nt; nt++) { // col of in1
                for (uint32_t kt = 0; kt < Kt; kt++) { // col of in0, row of in1
                    // Read A's tile at (mt, kt)
                    if (mt % num_readers == reader_id) {
#ifdef ARCH_QUASAR
                        // Quasar: implicit-sync read. The DFB credit advances via the per-trid
                        // completion ISR; no reserve_back / barrier / push_back required.
                        noc.async_read<NocOptions::TXN_ID>(s0, dfb0, {.page_id = itileA}, {});
#else
                        dfb0.reserve_back(onetile);
                        noc.async_read(s0, dfb0, entry_size0, {.page_id = itileA}, {});
                        noc.async_read_barrier();
                        dfb0.push_back(onetile);
#endif
                    }

                    // Read B's tile at (kt, nt)
                    if (mt % num_readers == reader_id && kt % num_readers == reader_id) {
#ifdef ARCH_QUASAR
                        noc.async_read<NocOptions::TXN_ID>(s1, dfb1, {.page_id = itileB}, {});
#else
                        dfb1.reserve_back(onetile);
                        noc.async_read(s1, dfb1, entry_size1, {.page_id = itileB}, {});
                        noc.async_read_barrier();
                        dfb1.push_back(onetile);
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

    dfb0.finish();
    dfb1.finish();
}
