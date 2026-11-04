// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Implicit-write availability guard. Claims the single posted entry on every
// round-robin tile counter, then issues one more write that lands back on the
// first counter. That counter's only posted entry is already claimed by a write
// whose ACK is still outstanding, so the write must block until a new credit
// arrives -- hardware occupancy alone still reports the entry as available.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_tile_counters = get_arg(args::num_tile_counters);

    DataflowBuffer dfb(dfb::in);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::dst_tensor);

    Semaphore producer_ready(sem::producer_ready);
    Semaphore second_attempt(sem::second_attempt);
    Semaphore second_returned(sem::second_returned);
    Semaphore credit_released(sem::credit_released);

    producer_ready.wait_min(num_tile_counters);

    for (uint32_t i = 0; i < num_tile_counters; ++i) {
        noc.async_write<NocOptions::TXN_ID>(dfb, tensor_accessor, {}, {.page_id = i});
    }

    second_attempt.up(1);
    noc.async_write<NocOptions::TXN_ID>(dfb, tensor_accessor, {}, {.page_id = num_tile_counters});
    second_returned.up(1);

    credit_released.wait_min(1);
    noc.async_write_barrier();
}
