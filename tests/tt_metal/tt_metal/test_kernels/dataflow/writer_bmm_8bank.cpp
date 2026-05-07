// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#ifdef ARCH_QUASAR
#include "api/kernel_thread_globals.h"
#include "experimental/dataflow_buffer.h"
#endif

void kernel_main() {
    uintptr_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t writer_id = 0;
    uint32_t num_writers = 1;
    uint32_t batch_start = 0;
#ifdef ARCH_QUASAR
    writer_id = get_my_thread_id();
    num_writers = get_num_threads();
    batch_start = get_arg_val<uint32_t>(8);
#endif

    constexpr int onetile = 1;
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb(2);
    const uint32_t tile_bytes = dfb.get_entry_size();
#else
    constexpr uint32_t cb_id_out0 = 16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    experimental::CircularBuffer cb(cb_id_out0);
#endif

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    experimental::Noc noc;

    // C is MN so we iterate in tile RM order; batch_start offsets into the correct DRAM tile range
    uint32_t itileC = batch_start * Mt * Nt;
    for (uint32_t nb = 0; nb < batch; nb++) {
        // uint32_t itileC = itileC_batch;
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {  // output tile row of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {  // output tile col of C
                // bmm will generate C's tiles C=A*B, MN=MK*KN, in row major order, we just read them from CB and write
                // out to DRAM
#ifdef ARCH_QUASAR
                if (mt_C % num_writers == writer_id) {
                    dfb.wait_front(onetile);
                    uint32_t l1_read_addr = dfb.get_read_ptr();
                    noc_async_write_tile(itileC, s, l1_read_addr);
                    noc.async_write_barrier();
                    dfb.pop_front(onetile);
                }
#else
                cb.wait_front(onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                // Using legacy API because async_write_tile is not supported yet
                noc_async_write_tile(itileC, s, l1_read_addr);
                noc.async_write_barrier();
                cb.pop_front(onetile);
#endif
                // DEVICE_PRINT("WC{0} a{1}\n{0} {2}\n", itileC, dst_addr, uint32_t(dst_noc_addr));
                itileC++;
            }
        }
    }

#ifdef ARCH_QUASAR
    dfb.finish();
#endif
}
