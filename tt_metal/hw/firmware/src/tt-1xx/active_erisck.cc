// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "noc_parameters.h"
#include "internal/ethernet/dataflow_api.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "internal/risc_attribs.h"
#include "tensix.h"
#include "tensix_types.h"
#include "internal/ethernet/tt_eth_api.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "stream_io_map.h"
#include "tdma_xmov.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#include <stdint.h>

extern "C" [[gnu::section(".start")]]
void _start() {
#if !defined(ENABLE_2_ERISC_MODE)
    asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");
#endif
    extern uint32_t __kernel_data_lma[];
    do_crt1((uint32_t tt_l1_ptr*)__kernel_data_lma);

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    }

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
            ASSERT(ncrisc_noc_packet_tags_cleared(NOC_INDEX), DebugAssertNCriscNOCPacketTagClearedTripped);
            // Assert (watcher) then restore: a kernel that programmed a custom read VC must leave the read
            // command buffer at the firmware default, since the next kernel's plain reads inherit NOC_CTRL.
            // The reset is the always-on safety net; the assert blames the offending kernel before it runs.
            ASSERT(ncrisc_noc_read_vc_is_default(NOC_INDEX), DebugAssertNCriscNOCReadVCNotDefaultTripped);
            noc_reset_cmd_buf_vc_to_default(NOC_INDEX);
            WAYPOINT("NKFD");
        }

        // Ensure no eth transactions are outstanding
        for (uint32_t i = 0; i < eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS; i++) {
            ASSERT(erisc_info->channels[i].bytes_sent == 0);
        }
    }
}
