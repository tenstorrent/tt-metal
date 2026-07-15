// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Nt = get_arg(args::Nt);
    uint32_t batch = get_arg(args::batch);
    uint32_t batch_start = get_arg(args::batch_start);

    uint32_t writer_id = get_my_thread_id();
    uint32_t num_writers = get_num_threads();

    constexpr int onetile = 1;
    DataflowBuffer dfb(dfb::dst);
#ifndef ARCH_QUASAR
    const uint32_t entry_size = dfb.get_entry_size();
#endif

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;

    // C is MN so we iterate in tile RM order; batch_start offsets into the correct DRAM tile range
    uint32_t itileC = batch_start * Mt * Nt;
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {  // output tile row of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {  // output tile col of C
                if (mt_C % num_writers == writer_id) {
#ifdef ARCH_QUASAR
                    // Quasar: implicit-sync write. The DFB credit advances via the per-trid
                    // completion ISR; no wait_front / barrier / pop_front required.
                    noc.async_write<NocOptions::TXN_ID>(dfb, s, {}, {.page_id = itileC});
#else
                    dfb.wait_front(onetile);
                    noc.async_write(dfb, s, entry_size, {}, {.page_id = itileC});
                    noc.async_write_barrier();
                    dfb.pop_front(onetile);
#endif
                }
                itileC++;
            }
        }
    }

    dfb.finish();
}
