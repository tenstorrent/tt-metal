// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Power microbenchmark of a data movement RISC that waits (test_layer_energy.py::test_wait_power).
//
// Each mode runs for `cycles` wall-clock cycles, then returns:
//   1 = spin on the wall-clock register (riscv_wait),
//   2 = poll an L1 semaphore that never matches, as noc_semaphore_wait does (fence + L1 load per poll),
//   3 = the same poll with a `pause`-cycle riscv_wait between polls,
//   4 = loop 4 KB NoC reads of this core's own L1 (an active reference: NoC + L1 traffic).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t wall_clock() { return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t cycles = get_compile_time_arg_val(1);
    constexpr uint32_t pause = get_compile_time_arg_val(2);
    constexpr uint32_t buffer_offset = get_compile_time_arg_val(3);
    constexpr uint32_t never = 0xA5A5A5A5u;
    const uint32_t start = wall_clock();

    if constexpr (mode == 1) {
        while (wall_clock() - start < cycles) {
        }
    } else if constexpr (mode == 2 || mode == 3) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        do {
            for (uint32_t i = 0; i < 16; i++) {
                invalidate_l1_cache();
                if (*sem == never) {
                    return;
                }
                if constexpr (mode == 3) {
                    riscv_wait(pause);
                }
            }
        } while (wall_clock() - start < cycles);
    } else if constexpr (mode == 4) {
        constexpr uint32_t bytes = 4096;
        const uint32_t buffer = get_write_ptr(0) + buffer_offset;
        const uint64_t source = get_noc_addr(my_x[noc_index], my_y[noc_index], buffer);
        do {
            noc_async_read(source, buffer + bytes, bytes);
            noc_async_read_barrier();
        } while (wall_clock() - start < cycles);
    }
}
