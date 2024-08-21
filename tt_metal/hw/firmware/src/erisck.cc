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
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "noc_addr_ranges_gen.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include <kernel_includes.hpp>


uint8_t noc_index = NOC_INDEX;

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

void __attribute__((section("erisc_l1_code"))) kernel_launch() {
    DeviceZoneScopedMainChildN("ERISC-KERNEL");
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    kernel_main();
    mailboxes->launch.go.run = RUN_MSG_DONE;
    if (routing_info->routing_enabled and mailboxes->launch.kernel_config.mode == DISPATCH_MODE_DEV) {
        uint64_t dispatch_addr =
            NOC_XY_ADDR(NOC_X(mailboxes->launch.kernel_config.dispatch_core_x),
                        NOC_Y(mailboxes->launch.kernel_config.dispatch_core_y), DISPATCH_MESSAGE_ADDR);
        internal_::notify_dispatch_core_done(dispatch_addr);
    }
}
