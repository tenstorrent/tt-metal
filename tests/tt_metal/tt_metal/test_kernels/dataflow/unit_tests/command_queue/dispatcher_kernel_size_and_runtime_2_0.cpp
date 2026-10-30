// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 variant of dispatcher_kernel_size_and_runtime.cpp. KERNEL_RUNTIME_MICROSECONDS is a cycle
// count here, and since Gen2 requires a zero semaphore initial value, the semaphore check reads back
// zero and then leaves it non-zero.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

constexpr uint32_t num_unique_rt_args = get_arg(args::num_unique_rt_args);
constexpr uint32_t num_common_rt_args = get_arg(args::num_common_rt_args);
constexpr uint32_t unique_rt_args_vals_offset = get_arg(args::unique_rt_args_vals_offset);
constexpr uint32_t common_rt_args_vals_offset = get_arg(args::common_rt_args_vals_offset);
constexpr uint32_t num_sems = get_arg(args::num_sems);

#if KERNEL_SIZE_BYTES > 16
constexpr uint32_t empty_kernel_bytes = 16;
[[gnu::section(".text"), gnu::used]]
static uint8_t lorem_ipsum[KERNEL_SIZE_BYTES - empty_kernel_bytes];
#endif

static inline uint32_t read_cycle_count() {
    uint32_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// DFB accessor names are compile-time tokens, so each index needs its own statement.
#define VERIFY_DFB(idx)                                                                   \
    {                                                                                     \
        DataflowBuffer dfb(dfb::dfb_##idx);                                               \
        const uint32_t expected_size = get_vararg(num_unique_rt_args + num_sems + (idx)); \
        if (dfb.get_entry_size() != expected_size) {                                      \
            DPRINT(                                                                       \
                "{} Actual dataflow buffer entry size: {} Expected entry size: {}\n",     \
                (idx),                                                                    \
                dfb.get_entry_size(),                                                     \
                expected_size);                                                           \
            ASSERT(0);                                                                    \
            while (true); /* Hang kernel if values aren't correct */                      \
        }                                                                                 \
    }

#define VERIFY_SEM(idx)                                                              \
    {                                                                                \
        Semaphore s(sem::sem_##idx);                                                 \
        if (s.value() != 0) {                                                        \
            DPRINT("{} Actual semaphore value: {} Expected: 0\n", (idx), s.value()); \
            ASSERT(0);                                                               \
            while (true); /* Hang kernel if values aren't correct */                 \
        }                                                                            \
        s.up(1);                                                                     \
    }

void kernel_main() {
    // Unsigned subtraction so the comparison stays correct across a counter wrap.
    const uint32_t start_time = read_cycle_count();
    while (read_cycle_count() - start_time < KERNEL_RUNTIME_MICROSECONDS);

    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        const uint32_t rt_arg = get_vararg(i);
        const uint32_t expected = i + unique_rt_args_vals_offset;
        if (rt_arg != expected) {
            DPRINT("Actual runtime argument value: {} Expected runtime argument value: {}\n", rt_arg, expected);
            ASSERT(0);
            while (true);  // Hang kernel if values aren't correct
        }
    }

#if NUM_TEST_DFBS > 0
    VERIFY_DFB(0)
#endif
#if NUM_TEST_DFBS > 1
    VERIFY_DFB(1)
#endif
#if NUM_TEST_DFBS > 2
    VERIFY_DFB(2)
#endif
#if NUM_TEST_DFBS > 3
    VERIFY_DFB(3)
#endif
#if NUM_TEST_DFBS > 4
    VERIFY_DFB(4)
#endif
#if NUM_TEST_DFBS > 5
    VERIFY_DFB(5)
#endif
#if NUM_TEST_DFBS > 6
    VERIFY_DFB(6)
#endif
#if NUM_TEST_DFBS > 7
    VERIFY_DFB(7)
#endif
#if NUM_TEST_DFBS > 8
    VERIFY_DFB(8)
#endif
#if NUM_TEST_DFBS > 9
    VERIFY_DFB(9)
#endif
#if NUM_TEST_DFBS > 10
    VERIFY_DFB(10)
#endif
#if NUM_TEST_DFBS > 11
    VERIFY_DFB(11)
#endif
#if NUM_TEST_DFBS > 12
    VERIFY_DFB(12)
#endif
#if NUM_TEST_DFBS > 13
    VERIFY_DFB(13)
#endif
#if NUM_TEST_DFBS > 14
    VERIFY_DFB(14)
#endif
#if NUM_TEST_DFBS > 15
    VERIFY_DFB(15)
#endif

#if NUM_TEST_SEMS > 0
    VERIFY_SEM(0)
#endif
#if NUM_TEST_SEMS > 1
    VERIFY_SEM(1)
#endif
#if NUM_TEST_SEMS > 2
    VERIFY_SEM(2)
#endif
#if NUM_TEST_SEMS > 3
    VERIFY_SEM(3)
#endif
#if NUM_TEST_SEMS > 4
    VERIFY_SEM(4)
#endif
#if NUM_TEST_SEMS > 5
    VERIFY_SEM(5)
#endif
#if NUM_TEST_SEMS > 6
    VERIFY_SEM(6)
#endif
#if NUM_TEST_SEMS > 7
    VERIFY_SEM(7)
#endif
#if NUM_TEST_SEMS > 8
    VERIFY_SEM(8)
#endif
#if NUM_TEST_SEMS > 9
    VERIFY_SEM(9)
#endif
#if NUM_TEST_SEMS > 10
    VERIFY_SEM(10)
#endif
#if NUM_TEST_SEMS > 11
    VERIFY_SEM(11)
#endif
#if NUM_TEST_SEMS > 12
    VERIFY_SEM(12)
#endif
#if NUM_TEST_SEMS > 13
    VERIFY_SEM(13)
#endif
#if NUM_TEST_SEMS > 14
    VERIFY_SEM(14)
#endif
#if NUM_TEST_SEMS > 15
    VERIFY_SEM(15)
#endif

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        const uint32_t common_rt_arg = get_common_vararg(i);
        const uint32_t expected = i + common_rt_args_vals_offset;
        if (common_rt_arg != expected) {
            DPRINT(
                "Actual common runtime argument value: {} Expected common runtime argument value: {}\n",
                common_rt_arg,
                expected);
            ASSERT(0);
            while (true);  // Hang kernel if values aren't correct
        }
    }
}
