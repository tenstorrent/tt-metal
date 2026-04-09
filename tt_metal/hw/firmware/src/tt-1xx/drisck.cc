// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "internal/tt-1xx/risc_common.h"
#include "noc.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/stack_usage.h"

#include <kernel_includes.hpp>

extern "C" [[gnu::section(".start")]]
uint32_t _start() {
    // Enable GPREL optimizations.
    asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");
    mark_stack_usage();
    extern uint32_t __kernel_data_lma[];
    do_crt1((uint32_t tt_l1_ptr*)__kernel_data_lma);

    noc_local_state_init(NOC_INDEX);

    {
        DeviceZoneScopedMainChildN("DRISC-KERNEL");
        WAYPOINT("K");
        kernel_main();
        WAYPOINT("KD");
        // DRAM kernels always use dedicated NOC — assert all transactions have landed.
        WAYPOINT("NKFW");
        ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
        ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCNonpostedWritesSentTripped);
        ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX), DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
        ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCPostedWritesSentTripped);
        WAYPOINT("NKFD");
    }
    return measure_stack_usage();
}
