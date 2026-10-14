// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program A, BRISC on the congestor cores, same NoC as the A cores. For FLOOD_CYCLES it streams
// CHUNK_BYTES writes and/or atomics at one destination core, then drains its own traffic.
// KIND: 0 = writes, 1 = atomics, 2 = both. Writes and atomics share NOC_UNICAST_WRITE_VC with A's atomic.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "late_atomic_common.h"

void kernel_main() {
    constexpr uint32_t FLOOD_CYCLES = get_compile_time_arg_val(0);
    constexpr uint32_t CHUNK_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t KIND = get_compile_time_arg_val(2);

    const uint32_t scratch_addr = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_x = get_arg_val<uint32_t>(2);
    const uint32_t dst_y = get_arg_val<uint32_t>(3);

    const uint64_t dst_data = get_noc_addr(dst_x, dst_y, scratch_addr);
    const uint64_t dst_sem = get_noc_addr(dst_x, dst_y, stats_addr + S_SEM_FLOOD * sizeof(uint32_t));

    const uint64_t t_end = c_tensix_core::read_wall_clock() + FLOOD_CYCLES;
    do {
        for (uint32_t k = 0; k < 8; ++k) {
            if constexpr (KIND != 1) {
                noc_async_write(scratch_addr, dst_data, CHUNK_BYTES);
            }
            if constexpr (KIND != 0) {
                noc_semaphore_inc(dst_sem, 1);
            }
        }
    } while (c_tensix_core::read_wall_clock() < t_end);

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
