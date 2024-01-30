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
#include <kernel.cpp>


uint8_t noc_index = NOC_INDEX;

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

void __attribute__((section("erisc_l1_code"))) kernel_launch() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    kernel_profiler::mark_kernel_start();
    kernel_main();
    uint64_t dispatch_addr = NOC_XY_ADDR(NOC_X(DISPATCH_CORE_X), NOC_Y(DISPATCH_CORE_Y), DISPATCH_MESSAGE_ADDR);
    internal_::notify_dispatch_core_done(dispatch_addr);
    erisc_info->launch_user_kernel = 0;
    kernel_profiler::mark_kernel_end();
}
