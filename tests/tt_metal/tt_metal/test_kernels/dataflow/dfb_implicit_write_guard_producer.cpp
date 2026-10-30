// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Credit controller for the implicit-write availability guard. Posts exactly one
// entry on each round-robin tile counter, then holds the next credit back long
// enough to observe whether the consumer's implicit write returned without it.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

namespace {

// Bounded so a consumer that correctly blocks cannot deadlock the controller.
constexpr uint32_t kObservationSpins = 65536;

}  // namespace

void kernel_main() {
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::out);
    Semaphore producer_ready(sem::producer_ready);
    Semaphore second_attempt(sem::second_attempt);
    Semaphore second_returned(sem::second_returned);
    Semaphore credit_released(sem::credit_released);

    // One entry on this thread's tile counter: enough for the consumer's first
    // write on this counter and nothing more.
    dfb.reserve_back(1);
    dfb.push_back(1);
    producer_ready.up(1);

    second_attempt.wait_min(1);

    if (get_my_thread_id() == 0) {
        bool returned_before_credit = false;
        for (uint32_t spin = 0; spin < kObservationSpins; ++spin) {
            if (second_returned.value() != 0) {
                returned_before_credit = true;
                break;
            }
            asm volatile("" ::: "memory");
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr + MEM_L1_UNCACHED_BASE) =
            returned_before_credit ? 1u : 0u;

        // Release the credit the consumer's blocked write is waiting on.
        dfb.reserve_back(1);
        dfb.push_back(1);
        credit_released.up(1);
    }

    second_returned.wait_min(1);
}
