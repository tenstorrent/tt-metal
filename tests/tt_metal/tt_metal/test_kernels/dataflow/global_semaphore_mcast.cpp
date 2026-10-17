// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore NoC multicast to the rectangle [start, end] (virtual coordinates, sender outside it).
// GLOBAL_SEM_INC: inc_multicast() by `value`, twice. Otherwise: set(value) locally, then set_multicast()
// copies it to the rectangle.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_x = get_arg(args::start_x);
    const uint32_t start_y = get_arg(args::start_y);
    const uint32_t end_x = get_arg(args::end_x);
    const uint32_t end_y = get_arg(args::end_y);
    const uint32_t num_dests = get_arg(args::num_dests);
    const uint32_t value = get_arg(args::value);

    GlobalSemaphore flag(sem::flag);
    Noc noc;
#ifdef GLOBAL_SEM_INC
    flag.inc_multicast(noc, start_x, start_y, end_x, end_y, value, num_dests);
    flag.inc_multicast(noc, start_x, start_y, end_x, end_y, value, num_dests);
    noc.async_atomic_barrier();
#else
    flag.set(value);
    flag.set_multicast(noc, start_x, start_y, end_x, end_y, num_dests);
    noc.async_write_barrier();
#endif
}
