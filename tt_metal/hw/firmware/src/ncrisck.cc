// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tensix_functions.h"
#include "c_tensix_core.h"

#include "kernel_includes.hpp"


uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
uint32_t noc_posted_writes_num_issued[NUM_NOCS];

extern uint32_t __kernel_init_local_l1_base[];

void kernel_launch() {

  DeviceZoneScopedMainChildN("NCRISC-KERNEL");
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < KERNEL_RUN_TIME);
#endif
#else
#ifdef ARCH_BLACKHOLE
    firmware_kernel_common_init((void tt_l1_ptr *)__kernel_init_local_l1_base);
#else
    firmware_kernel_common_init((void tt_l1_ptr *)(MEM_NCRISC_INIT_IRAM_L1_BASE + (uint32_t)__kernel_init_local_l1_base - MEM_NCRISC_IRAM_BASE));
#endif

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    } else {
        noc_local_state_init(NOC_0);
        noc_local_state_init(NOC_1);
    }

    kernel_main();
#endif
}
