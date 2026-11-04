// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// All-reduce reducer core, BRISC: sends the reduced chunk to the staging buffers of the two communication cores (they
// multicast it to the other chips) and to the landing core of its tile columns on this chip.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_STG = get_compile_time_arg_val(1);
    constexpr uint32_t Q = get_compile_time_arg_val(2);
    constexpr uint32_t NP = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_STG0 = get_compile_time_arg_val(4);
    constexpr uint32_t CHUNK = Q * 512;

    const uint32_t gsem = get_common_arg_val<uint32_t>(0);
    const uint32_t land_base = get_common_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    const uint32_t fx = get_arg_val<uint32_t>(2);  // forward / backward communication core of this reducer's link
    const uint32_t fy = get_arg_val<uint32_t>(3);
    const uint32_t bx = get_arg_val<uint32_t>(4);
    const uint32_t by = get_arg_val<uint32_t>(5);
    const uint32_t lx = get_arg_val<uint32_t>(6 + 2 * c * NP + 2 * p);
    const uint32_t ly = get_arg_val<uint32_t>(7 + 2 * c * NP + 2 * p);

    {
        DeviceZoneScopedN("W_WAITOUT");
        cb_wait_front(CB_OUT, Q);
    }
    const uint32_t src = get_read_ptr(CB_OUT);
    const uint32_t stg = get_write_ptr(CB_STG) + p * CHUNK;
    noc_async_write(src, get_noc_addr(fx, fy, stg), CHUNK);
    noc_async_write(src, get_noc_addr(bx, by, stg), CHUNK);
    noc_async_write(src, get_noc_addr(lx, ly, land_base), CHUNK);
    noc_async_write_barrier();
    const uint32_t sem_addr = get_semaphore(SEM_STG0 + p);
    noc_semaphore_inc(get_noc_addr(fx, fy, sem_addr), 1);
    noc_semaphore_inc(get_noc_addr(bx, by, sem_addr), 1);
    noc_semaphore_inc(get_noc_addr(lx, ly, gsem), 1);
    noc_async_atomic_barrier();
    cb_pop_front(CB_OUT, Q);
}
