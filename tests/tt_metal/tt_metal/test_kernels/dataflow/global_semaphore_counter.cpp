// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Many-to-one GlobalSemaphore counter. GLOBAL_SEM_SENDER: up() the hub's copy of sem::counter
// `increments` times over the NoC. Otherwise (the hub): wait for `expected`, report the value, then
// drain it with down() (GLOBAL_SEM_USE_DOWN) or set(0), and report the value again.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef GLOBAL_SEM_SENDER
    const uint32_t hub_x = get_arg(args::hub_x);
    const uint32_t hub_y = get_arg(args::hub_y);
    const uint32_t increments = get_arg(args::increments);

    GlobalSemaphore counter(sem::counter);
    Noc noc;
    for (uint32_t i = 0; i < increments; i++) {
        counter.up(noc, hub_x, hub_y, 1);
    }
    noc.async_atomic_barrier();
#else
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t expected = get_arg(args::expected);

    GlobalSemaphore counter(sem::counter);
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
#ifdef GLOBAL_SEM_USE_DOWN
    counter.down(expected);
    report[0] = expected;
#else
    counter.wait(expected);
    report[0] = counter.value();
    counter.set(0);
#endif
    report[1] = counter.value();
#endif
}
