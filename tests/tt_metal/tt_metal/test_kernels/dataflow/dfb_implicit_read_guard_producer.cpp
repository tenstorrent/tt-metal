// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Implicit-read availability guard. The counters are preloaded so exactly one
// ring slot is free on every round-robin tile counter. One implicit read
// reserves each of those slots. Because those reads have not posted yet,
// hardware free space still reports one free slot when the next read revisits
// the first counter, so that read must block until the consumer frees a slot.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/dataflow_buffer_test_helpers.h"

void kernel_main() {
    constexpr uint32_t preload_posted = get_arg(args::preload_posted);
    constexpr uint32_t num_tile_counters = get_arg(args::num_tile_counters);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    Semaphore producer_ready(sem::producer_ready);
    Semaphore consumer_ready(sem::consumer_ready);
    Semaphore second_attempt(sem::second_attempt);
    Semaphore second_returned(sem::second_returned);
    Semaphore credit_released(sem::credit_released);

    // Rendezvous as in the D1 wrap test: neither side may issue traffic until
    // both POSTED and ACKED have been preloaded.
    preload_posted_counter(dfb, preload_posted);
    producer_ready.up(1);
    // Thread 0 signals only after every consumer counter has been preloaded.
    consumer_ready.wait_min(1);

    for (uint32_t i = 0; i < num_tile_counters; ++i) {
        noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = i}, {});
    }

    second_attempt.up(1);
    noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = num_tile_counters}, {});
    second_returned.up(1);

    credit_released.wait_min(1);
    noc.async_read_barrier();
}
