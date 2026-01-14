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
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/stack_usage.h"
#include <kernel_includes.hpp>
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "api/remote_circular_buffer.h"
#endif
#ifdef UDM_MODE
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#endif

namespace ckernel {
// Transition shim
#if defined(__PTR_CONST)
#define PTR_CONST const
#else
#define PTR_CONST
#endif
volatile tt_reg_ptr uint* PTR_CONST regfile = reinterpret_cast<volatile uint*>(REGFILE_BASE);
volatile tt_reg_ptr uint* PTR_CONST pc_buf_base = reinterpret_cast<volatile uint*>(PC_BUF_BASE);

// There are 16 mailboxes within each Tensix tile, one from each of RISCV (B, T0, T1, T2) to each of RISCV (B, T0, T1,
// T2). Note that there are no mailboxes to or from RISCV NC. The 16 mailboxes are referenced using 4 particular address
// ranges starting from bases listed below. Which particular mailbox is being referenced depends on the address, the
// issuing RISCV, and whether the access is a read or a write.
volatile tt_reg_ptr uint* PTR_CONST mailbox_base[4] = {
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX0_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX2_BASE),
    reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_MAILBOX3_BASE)};
#undef PTR_CONST
}  // namespace ckernel

extern "C" [[gnu::section(".start")]]
uint32_t _start() {
    // Enable GPREL optimizations.
    asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");
    mark_stack_usage();
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
    wait_for_go_message();
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    extern uint32_t __kernel_data_lma[];
    do_crt1((uint32_t tt_l1_ptr*)__kernel_data_lma);

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
    {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        EARLY_RETURN_FOR_DEBUG
        WAYPOINT("K");
        kernel_main();
        WAYPOINT("KD");
        if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
            WAYPOINT("NKFW");
            // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and the
            // NOC interface is in a known idle state for the next kernel. Dispatch kernels don't increment noc counters
            // so we only include this for non-dispatch kernels
            ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
            ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCNonpostedWritesSentTripped);
            ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX), DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
            ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCPostedWritesSentTripped);
            WAYPOINT("NKFD");
        }
    }
    EARLY_RETURN_FOR_DEBUG_EXIT;
#endif
    return measure_stack_usage();
}
