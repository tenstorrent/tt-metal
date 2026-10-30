// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore NoC multicast to the rectangle [start, end] (virtual coordinates, sender outside it).
// The semaphore's address is common runtime arg 0. Runtime args {start_x, start_y, end_x, end_y, num_dests, value}.
// GLOBAL_SEM_INC: inc_multicast() by `value`, twice. Otherwise: set(value) locally, then set_multicast()
// copies it to the rectangle.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"

constexpr GlobalSemaphoreBindingToken flag_token{0};

void kernel_main() {
    const uint32_t start_x = get_arg_val<uint32_t>(0);
    const uint32_t start_y = get_arg_val<uint32_t>(1);
    const uint32_t end_x = get_arg_val<uint32_t>(2);
    const uint32_t end_y = get_arg_val<uint32_t>(3);
    const uint32_t num_dests = get_arg_val<uint32_t>(4);
    const uint32_t value = get_arg_val<uint32_t>(5);

    GlobalSemaphore flag(flag_token);
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
