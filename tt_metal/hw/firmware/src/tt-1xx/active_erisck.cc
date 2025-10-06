// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
void _start() {
    extern uint32_t __kernel_data_lma[];
    asm("nop; nop; nop;");
    // asm(".rept 512; nop; .endr");

    // do_crt1((uint32_t tt_l1_ptr*)__kernel_data_lma);
    // #pragma gcc unroll 0
    //     for (uint32_t i = 0; i < 1; ++i) {
    //         asm volatile(
    //             "li t0, 0xFFB00000\n\t"     // local_mem = 0xFFB00000
    //             "lw t1, 0(t0)\n\t"          // local_mem[0]
    //             "lw t2, 16(t0)\n\t"         // local_mem[4]
    //             "lw t3, 32(t0)\n\t"         // local_mem[8]
    //             :
    //             :
    //             : "t0", "t1", "t2", "t3"
    //         );
    //     }

    // noc_local_state_init(NOC_INDEX);

    // {
    //     DeviceZoneScopedMainChildN("ERISC-KERNEL");
    //     WAYPOINT("K");
    // kernel_main();
    //     WAYPOINT("KD");
    //     if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
    //         WAYPOINT("NKFW");
    //         // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and
    //         the
    //         // NOC interface is in a known idle state for the next kernel.
    //         ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
    //         ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), DebugAssertNCriscNOCNonpostedWritesSentTripped);
    //         ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX),
    //         DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped); ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX),
    //         DebugAssertNCriscNOCPostedWritesSentTripped); WAYPOINT("NKFD");
    //     }

    //     // Ensure no eth transactions are outstanding
    //     for (uint32_t i = 0; i < eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS; i++) {
    //         ASSERT(erisc_info->channels[i].bytes_sent == 0);
    //     }
    // }
}
