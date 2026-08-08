// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Remote-up probe for the scoped Semaphore class: exact-count coverage of
// Semaphore::up(noc, x, y, v). One source, two roles via -D (mirrors the REMOTE_TARGET
// pattern in noc_self_atomic.cpp):
//   REMOTE_SENDER (off-node): every thread bumps sem::counter on the semaphore's node,
//                             increment_times times, through the class's remote up().
//   receiver (default, on the semaphore's node): waits for the exact expected total, then
//                             reports the baked scope and value(). OBSERVE binding, so the
//                             accessor is read-only -- it only waits and reads.
// Scope is host-baked and picked up via CTAD, so the same source runs under any scope.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef REMOTE_SENDER
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);

    Semaphore counter(sem::counter);  // CTAD deduces the host-baked scope
    Noc noc;
    for (uint32_t i = 0; i < increment_times; i++) {
        counter.up(noc, remote_noc_x, remote_noc_y, 1);
    }
    // Remote up() issues a non-posted NoC atomic without draining it (unlike local EXTERNAL
    // up(), which fences internally); drain this hart's atomics before exit.
    noc.async_atomic_barrier();
#else
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t expected = get_arg(args::expected);

    Semaphore counter(sem::counter);  // CTAD deduces the scope AND the OBSERVE read-only bit
    counter.wait_min(expected);

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = static_cast<uint32_t>(sem::counter.scope);
    report[1] = counter.value();
#if defined(ARCH_QUASAR)
    // Make the report visible to the host readback of TL1.
    flush_l2_cache_line(report_addr);
    flush_l2_cache_line(report_addr + sizeof(uint32_t));
#endif
#endif
}
