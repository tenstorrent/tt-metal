// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PrefetcherPipe receiver that times delivery. It takes its entries `pop_batch` at a time, as a consumer that works on
// several blocks at once does, records this core's cycle count when each batch has arrived, and pops the batch unread.
// Stamping once per batch keeps this receiver's own per-entry cost out of the pace it measures. At the end it writes
// the stamps to its page of the output tensor.
//
// The stamps are this core's own cycle counter (get_timestamp_32b), so only stamps from one receiver can be
// subtracted from each other.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t pop_batch = get_arg(args::pop_batch);
    // This receiver's page (row) of the output tensor.
    const uint32_t out_page = get_arg_val<uint32_t>(0);

    const auto out = TensorAccessor(tensor::arrivals);
    Scratchpad<uint32_t> scratchpad(scratch::stamps);
    const uint32_t stamps_addr =
        static_cast<uint32_t>(local_mem::detail::AddressAccess::address(scratchpad.local_mem()));
    // The NoC reads the stamps out of L1 below, and on Quasar it does not snoop the data-movement caches, so the
    // stamps are stored through the uncached alias.
#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* stamps =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamps_addr + MEM_L1_UNCACHED_BASE);
#else
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamps_addr);
#endif

    constexpr uint32_t num_batches = (num_entries + pop_batch - 1) / pop_batch;
    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::in);
    for (uint32_t b = 0, e = 0; b < num_batches; ++b) {
        const uint32_t batch = num_entries - e < pop_batch ? num_entries - e : pop_batch;
        pipe.wait_front(batch);
        stamps[b] = get_timestamp_32b();
        pipe.pop_front(batch, noc);
        e += batch;
    }

    noc_async_write(stamps_addr, out.get_noc_addr(out_page), num_batches * sizeof(uint32_t));
    noc_async_write_barrier();
}
