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
#include "stream_io_map.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "internal/hw_thread.h"
#include "hostdev/dev_msgs.h"
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
    // TODO: initialize globals and bss
    uint32_t hartid = internal_::get_hw_thread_idx();

    // Obtain launch message from mailbox and derive thread 0 (lowest hartid with same kernel).
    uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    launch_msg_t tt_l1_ptr* launch_msg = &(*GET_MAILBOX_ADDRESS_DEV(launch))[launch_idx];
    uint32_t my_kt = launch_msg->kernel_config.kernel_text_offset[hartid];
    uint32_t thread_0_hartid = MaxDMProcessorsPerCoreType;
    for (uint32_t j = 0; j < MaxDMProcessorsPerCoreType; j++) {
        if (launch_msg->kernel_config.kernel_text_offset[j] == my_kt) {
            thread_0_hartid = j;
            break;
        }
    }

    extern uint32_t __tdata_lma[];
    extern uint32_t __ldm_tdata_start[];
    extern uint32_t __ldm_tdata_end[];

    if (hartid == thread_0_hartid) {
        do_crt1(&__tdata_lma[__ldm_tdata_end - __ldm_tdata_start]);
        (*GET_MAILBOX_ADDRESS_DEV(shared_globals_ready))[hartid] = SHARED_GLOBALS_READY_GO;
    }

    do_thread_crt1(__tdata_lma);

    // Wait until first thread in the group has set its slot to GO.
    while ((*GET_MAILBOX_ADDRESS_DEV(shared_globals_ready))[thread_0_hartid] != SHARED_GLOBALS_READY_GO) {
    }

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

        // Setup after the go signal so the previous kernel has completed.
        num_kernel_threads = launch_msg->kernel_config.num_kernel_threads[hartid];
        my_thread_id = launch_msg->kernel_config.kernel_thread_id[hartid];

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
