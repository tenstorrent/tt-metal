// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Many-to-one GlobalSemaphore counter; the semaphore's address is common runtime arg 0.
// GLOBAL_SEM_SENDER: runtime args {hub_x, hub_y, increments}; up() the hub's copy `increments` times over the
// NoC. Otherwise (the hub): runtime args {report_addr, expected}; wait for `expected`, report the value, then
// drain it with down() (GLOBAL_SEM_USE_DOWN) or set(0), and report the value again.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"

constexpr GlobalSemaphoreBindingToken counter_token{0};

void kernel_main() {
#ifdef GLOBAL_SEM_SENDER
    const uint32_t hub_x = get_arg_val<uint32_t>(0);
    const uint32_t hub_y = get_arg_val<uint32_t>(1);
    const uint32_t increments = get_arg_val<uint32_t>(2);

    GlobalSemaphore counter(counter_token);
    Noc noc;
    for (uint32_t i = 0; i < increments; i++) {
        counter.up(noc, hub_x, hub_y, 1);
    }
    noc.async_atomic_barrier();
#else
    const uint32_t report_addr = get_arg_val<uint32_t>(0);
    const uint32_t expected = get_arg_val<uint32_t>(1);

    GlobalSemaphore counter(counter_token);
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
