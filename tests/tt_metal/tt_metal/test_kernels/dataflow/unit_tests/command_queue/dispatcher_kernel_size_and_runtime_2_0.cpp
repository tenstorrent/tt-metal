// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 variant of dispatcher_kernel_size_and_runtime.cpp: same checks, named compile-time args.
// Gen2 differences:
//   - spins on the RISC-V cycle counter, since c_tensix_core's wall clock is Gen1-only, so
//     KERNEL_RUNTIME_MICROSECONDS is a cycle count here rather than microseconds
//   - the Gen1 circular buffers become dataflow buffers, so the page-size check becomes an
//     entry-size check. Gen2 rejects a data-movement kernel bound as both ends of a DFB, so this
//     kernel takes the producer end and a blank compute kernel alongside it takes the consumer end
//   - the semaphore value is not checked: Gen2 rejects a non-zero initial value, so the Gen1
//     "reads back SEM_VAL" assertion has no equivalent, and checking for zero would pass whether
//     or not the semaphore was ever placed

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
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

// DFB accessor names are compile-time tokens, so each possible index needs its own statement. The
// host binds exactly NUM_TEST_DFBS of them and defines the count to match. The expected entry sizes
// follow the unique args and the semaphore ids in the runtime args, matching the host's order.
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
