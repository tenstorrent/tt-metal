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
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/tensix_functions.h"
#include "c_tensix_core.h"
#include "kernel_includes.hpp"
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "api/remote_circular_buffer.h"
#endif
#include "internal/debug/stack_usage.h"
#ifdef UDM_MODE
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#endif

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
uint32_t noc_posted_writes_num_issued[NUM_NOCS];

#if defined(ARCH_WORMHOLE)
extern "C" uint32_t kernel_launch(uint32_t offset) {
#else
extern "C" [[gnu::section(".start")]]
uint32_t _start() {
    uint32_t offset = 0;
#endif
    // Enable GPREL optimizations.
    asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");

    mark_stack_usage();
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
    wait_for_go_message();
    DeviceZoneScopedMainChildN("NCRISC-KERNEL");
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    extern uint32_t __kernel_data_lma[];
    do_crt1((uint32_t tt_l1_ptr*)((uint32_t)&__kernel_data_lma[0] + offset));

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    }
#ifdef UDM_MODE
    tt::tt_fabric::udm::fabric_local_state_init();
#endif
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
    wait_for_go_message();
    DeviceZoneScopedMainChildN("NCRISC-KERNEL");
    EARLY_RETURN_FOR_DEBUG
    WAYPOINT("K");
    kernel_main();
    WAYPOINT("KD");
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
    EARLY_RETURN_FOR_DEBUG_EXIT;
#endif
    return measure_stack_usage();
}
