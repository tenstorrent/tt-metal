// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 variant of random_program.cpp. Gen2 requires a zero semaphore initial value, so the
// semaphore check reads back zero and then leaves it non-zero for the next dispatch to reset.

#include <cstdint>

#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#endif

#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

constexpr uint32_t outer_loop = get_arg(args::outer_loop);
constexpr uint32_t middle_loop = get_arg(args::middle_loop);
constexpr uint32_t inner_loop = get_arg(args::inner_loop);
constexpr uint32_t num_unique_rt_args = get_arg(args::num_unique_rt_args);
constexpr uint32_t num_common_rt_args = get_arg(args::num_common_rt_args);
constexpr uint32_t entry_size_step = get_arg(args::entry_size_step);

// g_dfb_interface only exists on the unpack and pack TRISC images.
#if !defined(COMPILE_FOR_TRISC) || defined(TRISC_UNPACK)
#define RUN_CHECKS 1
#else
#define RUN_CHECKS 0
#endif

// DFB accessor names are compile-time tokens, so each index needs its own statement.
#define VERIFY_DFB(idx)                                                                                       \
    {                                                                                                         \
        DataflowBuffer dfb(dfb::dfb_##idx);                                                                   \
        const uint32_t expected = ((idx) + 1) * entry_size_step;                                              \
        if (dfb.get_entry_size() != expected) {                                                               \
            DPRINT("Problem with DFB idx: {} Expected: {} Got: {}\n", (idx), expected, dfb.get_entry_size()); \
            while (true); /* Purposefully hang the kernel if DFBs did not arrive correctly */                 \
        }                                                                                                     \
    }

#define VERIFY_SEM(idx)                                                                             \
    {                                                                                               \
        Semaphore s(sem::sem_##idx);                                                                \
        if (s.value() != 0) {                                                                       \
            DPRINT("Problem with Sem idx: {} Expected: 0 Got: {}\n", (idx), s.value());             \
            while (true); /* Purposefully hang the kernel if semaphores did not arrive correctly */ \
        }                                                                                           \
        s.up(1);                                                                                    \
    }

void kernel_main() {
#if RUN_CHECKS && NUM_TEST_DFBS > 0
    VERIFY_DFB(0)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 1
    VERIFY_DFB(1)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 2
    VERIFY_DFB(2)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 3
    VERIFY_DFB(3)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 4
    VERIFY_DFB(4)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 5
    VERIFY_DFB(5)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 6
    VERIFY_DFB(6)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 7
    VERIFY_DFB(7)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 8
    VERIFY_DFB(8)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 9
    VERIFY_DFB(9)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 10
    VERIFY_DFB(10)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 11
    VERIFY_DFB(11)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 12
    VERIFY_DFB(12)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 13
    VERIFY_DFB(13)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 14
    VERIFY_DFB(14)
#endif
#if RUN_CHECKS && NUM_TEST_DFBS > 15
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

#if RUN_CHECKS
    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        const uint32_t rt_arg = get_vararg(i);
        if (rt_arg != i) {
            DPRINT("Problem with unique RT Arg idx: {} Expected: {} Got: {}\n", i, i, rt_arg);
            while (true);  // Purposefully hang the kernel if Unique RT Args did not arrive correctly.
        }
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        const uint32_t rt_arg = get_common_vararg(i);
        const uint32_t expected = i + 100;
        if (rt_arg != expected) {
            DPRINT("Problem with common RT Arg idx: {} Expected: {} Got: {}\n", i, expected, rt_arg);
            while (true);  // Purposefully hang the kernel if Common RT Args did not arrive correctly.
        }
    }
#endif

    for (volatile uint32_t i = 0; i < outer_loop; i++) {
        for (volatile uint32_t j = 0; j < middle_loop; j++) {
            for (volatile uint32_t k = 0; k < inner_loop; k++) {
                // Do nothing
            }
        }
    }
}
