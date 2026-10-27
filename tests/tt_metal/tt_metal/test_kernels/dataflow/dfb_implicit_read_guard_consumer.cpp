// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Credit controller for the implicit-read availability guard. Preloads ACKED so
// the ring has exactly one free slot, then holds the next free slot back long
// enough to observe whether the producer's second implicit read returned early.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/dataflow_buffer_test_helpers.h"

namespace {

// Bounded so a producer that correctly blocks cannot deadlock the controller.
constexpr uint32_t kObservationSpins = 65536;

}  // namespace

void kernel_main() {
    constexpr uint32_t preload_acked = get_arg(args::preload_acked);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);
    Semaphore producer_ready(sem::producer_ready);
    Semaphore consumer_ready(sem::consumer_ready);
    Semaphore peer_preloaded(sem::peer_preloaded);
    Semaphore second_attempt(sem::second_attempt);
    Semaphore second_returned(sem::second_returned);
    Semaphore credit_released(sem::credit_released);

    // Each consumer thread owns one tile counter. Preload every counter, then let
    // thread 0 be the only writer of consumer_ready so the increment cannot be lost.
    if (preload_acked != 0) {
        preload_acked_counter(dfb, preload_acked);
    }
    if (get_num_threads() > 1 && get_my_thread_id() + 1 == get_num_threads()) {
        peer_preloaded.up(1);
    }
    if (get_my_thread_id() == 0) {
        if (get_num_threads() > 1) {
            peer_preloaded.wait_min(1);
        }
        consumer_ready.up(1);
    }
    producer_ready.wait_min(1);

    if (get_my_thread_id() != 0) {
        second_returned.wait_min(1);
        return;
    }

    second_attempt.wait_min(1);

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

    // Free the first counter, which is the one the next producer read revisits.
    dfb.pop_front(1);
    credit_released.up(1);

    second_returned.wait_min(1);
}
