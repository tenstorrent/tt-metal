// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tensix_functions.h"
#include "c_tensix_core.h"
#include "kernel_includes.hpp"
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "remote_circular_buffer_api.h"
#endif

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
uint32_t noc_posted_writes_num_issued[NUM_NOCS];

void kernel_launch(uint32_t kernel_base_addr) {
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
    global_program_barrier();
    DeviceZoneScopedMainChildN("NCRISC-KERNEL");
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    extern uint32_t __kernel_init_local_l1_base[];
    extern uint32_t __kernel_text_start[];
    do_crt1((uint32_t tt_l1_ptr*)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base -
                                  (uint32_t)__kernel_text_start));

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    }
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
// If OVERLAPPED_DISPATCH is defined, the kernel may start before the previous kernel finishes on all cores.
#if !defined(OVERLAPPED_DISPATCH) || (defined(DEBUG_EARLY_RETURN_KERNELS) && !defined(DISPATCH_KERNEL))
    global_program_barrier();
#endif
    DeviceZoneScopedMainChildN("NCRISC-KERNEL");
    EARLY_RETURN_FOR_DEBUG
    WAYPOINT("K");
    kernel_main();
    WAYPOINT("KD");
#ifdef OVERLAPPED_DISPATCH
    // Ensure that the previous kernel has completed before reporting this kernel as complete, to avoid mixing up the
    // done counter.
    global_program_barrier();
#endif
    // Checking is disabled on NCRISC for dispatch because dispatch_s, which
    // runs on NCRISC, does not track all transactions correctly.
#ifndef DISPATCH_KERNEL
    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        WAYPOINT("NKFW");
        // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and the NOC
        // interface is in a known idle state for the next kernel.
        ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
        ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCNonpostedWritesSentTripped);
        ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX), DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
        ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCPostedWritesSentTripped);
        WAYPOINT("NKFD");
    }
#endif
#endif
}
