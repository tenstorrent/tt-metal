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


CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
CQReadInterface cq_read_interface;

void ApplicationHandler(void) __attribute__((__section__(".init")));

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_xy_local_addr[NUM_NOCS];

uint8_t noc_index = NOC_INDEX;

void __attribute__((section("code_l1"))) risc_init();

void __attribute__((__section__("erisc_l1_code"))) kernel_launch() {
    RISC_POST_STATUS(0x12345678);
    kernel_main();
    RISC_POST_STATUS(0x12345679);
    disable_erisc_app();
}
void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    noc_init();

    risc_init();

    setup_cb_read_write_interfaces(0, 1, true, true);
    noc_local_state_init(noc_index);
    // TODO:: if bytes done not updated, yield to task 1
    kernel_launch();
}
