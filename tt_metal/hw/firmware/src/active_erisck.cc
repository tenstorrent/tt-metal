// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "noc_parameters.h"
#include "ethernet/dataflow_api.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "risc_attribs.h"
#include "tensix.h"
#include "tensix_types.h"
#include "tt_eth_api.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "stream_io_map.h"
#include "tdma_xmov.h"
#include "debug/dprint.h"
#include "debug/stack_usage.h"
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#include <stdint.h>

extern "C" [[gnu::section(".start")]]
uint32_t _start() {
    // Enable GPREL optimizations.
    asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");
    mark_stack_usage();
    extern uint32_t __kernel_data_lma[];
    do_crt1((uint32_t tt_l1_ptr*)__kernel_data_lma);

    noc_local_state_init(NOC_INDEX);

    {
        DeviceZoneScopedMainChildN("ERISC-KERNEL");
        WAYPOINT("K");
        kernel_main();
        WAYPOINT("KD");
        if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
            WAYPOINT("NKFW");
            // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and the
            // NOC interface is in a known idle state for the next kernel.
            ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
            ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCNonpostedWritesSentTripped);
            ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX), DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
            ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCPostedWritesSentTripped);
            WAYPOINT("NKFD");
        }

        // Ensure no eth transactions are outstanding
        for (uint32_t i = 0; i < eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS; i++) {
            ASSERT(erisc_info->channels[i].bytes_sent == 0);
        }
    }
    return measure_stack_usage();
}
