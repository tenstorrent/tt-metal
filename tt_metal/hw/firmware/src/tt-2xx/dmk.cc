// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/stack_usage.h"
#include <kernel_includes.hpp>
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "api/remote_circular_buffer.h"
#endif

extern "C" [[gnu::section(".start")]]
uint32_t _start() {
    // Enable GPREL optimizations.
    // asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");
    mark_stack_usage();
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
    wait_for_go_message();
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    // TODO: initilaize globals and bss
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    extern uint32_t __tdata_lma[];
    // for now this works for legacy kernels, we need to revisit this for new kernels
    // if (hartid == /* leading core */ 0) {
    extern uint32_t __ldm_tdata_start[];
    extern uint32_t __ldm_tdata_end[];
    do_crt1(&__tdata_lma[__ldm_tdata_end - __ldm_tdata_start]);
    // }
    do_thread_crt1(__tdata_lma);

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        // noc_local_state_init(NOC_INDEX); //TODO revisit this
    }
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
    wait_for_go_message();
    {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        EARLY_RETURN_FOR_DEBUG
        WAYPOINT("K");
        kernel_main();
        WAYPOINT("KD");
        if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
            WAYPOINT("NKFW");
            // TODO enable once NOC is ready
            // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and the
            // NOC interface is in a known idle state for the next kernel. Dispatch kernels don't increment noc counters
            // so we only include this for non-dispatch kernels
            // ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
            // ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCNonpostedWritesSentTripped);
            // ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX),
            // DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
            // ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCPostedWritesSentTripped);
            WAYPOINT("NKFD");
        }
    }
    EARLY_RETURN_FOR_DEBUG_EXIT;
#endif
    return measure_stack_usage();
}
