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
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>


uint8_t noc_index = NOC_INDEX;
constexpr uint32_t read_cmd_buf __attribute__((used)) = NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf __attribute__((used)) = NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf __attribute__((used)) = NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf __attribute__((used)) = NCRISC_AT_CMD_BUF;

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

void __attribute__((section("erisc_l1_code"))) kernel_launch() {
    DeviceZoneScopedMainChildN("ERISC-KERNEL");
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    kernel_main();
}
