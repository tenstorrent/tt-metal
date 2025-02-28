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
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#include <stdint.h>

void kernel_launch(uint32_t kernel_base_addr) {
    extern uint32_t __kernel_init_local_l1_base[];
    extern uint32_t __fw_export_end_text[];
    do_crt1((uint32_t tt_l1_ptr*)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base -
                                  (uint32_t)__fw_export_end_text));

    noc_local_state_init(NOC_INDEX);

    {
        DeviceZoneScopedMainChildN("ACTIVE-ERISC-KERNEL");
        kernel_main();
        if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
            WAYPOINT("NKFW");
            // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and the
            // NOC interface is in a known idle state for the next kernel.
            ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX));
            ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX));
            ASSERT(ncrisc_noc_nonposted_writes_flushed(NOC_INDEX));
            ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX));
            ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX));
            WAYPOINT("NKFD");
        } else {
            WAYPOINT("NKFW");
            // Assert that no noc transactions are outstanding, to ensure that all reads and writes have landed and the
            // NOC interface is in a known idle state for the next kernel.
            for (int noc = 0; noc < NUM_NOCS; noc++) {
                ASSERT(ncrisc_dynamic_noc_reads_flushed(noc));
                ASSERT(ncrisc_dynamic_noc_nonposted_writes_sent(noc));
                ASSERT(ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
                ASSERT(ncrisc_dynamic_noc_nonposted_atomics_flushed(noc));
                ASSERT(ncrisc_dynamic_noc_posted_writes_sent(noc));
            }
            WAYPOINT("NKFD");
        }
    }
}
