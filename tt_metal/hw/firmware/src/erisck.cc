// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "stream_io_map.h"
#include "tdma_xmov.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#include <stdint.h>

extern "C" void wzerorange(uint32_t *start, uint32_t *end);

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

extern "C" [[gnu::section(".start")]] void _start(uint32_t) {
    // Clear bss, we write to rtos_context_switch_ptr just below.
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    rtos_context_switch_ptr = (void (*)())RtosTable[0];
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

    while (mailboxes->go_message.signal != RUN_MSG_GO) {
        internal_::risc_context_switch();
        invalidate_l1_cache();
    }
    DeviceZoneScopedMainChildN("ERISC-KERNEL");

    kernel_main();
}
