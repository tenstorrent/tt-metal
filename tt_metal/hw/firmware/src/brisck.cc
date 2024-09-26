// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>

uint8_t noc_index = NOC_INDEX;
extern uint32_t __kernel_init_local_l1_base[];
uint8_t noc_mode = NOC_MODE;
const uint32_t read_cmd_buf __attribute__((used)) = NOC_MODE == DEDICATED_NOC_PER_DM ? BRISC_RD_CMD_BUF : MULTI_NOC_BRISC_RD_CMD_BUF;
const uint32_t write_cmd_buf __attribute__((used)) = NOC_MODE == DEDICATED_NOC_PER_DM ? BRISC_WR_CMD_BUF : MULTI_NOC_BRISC_WR_CMD_BUF;
const uint32_t write_reg_cmd_buf __attribute__((used)) = NOC_MODE == DEDICATED_NOC_PER_DM ? BRISC_WR_REG_CMD_BUF : MULTI_NOC_BRISC_WR_REG_CMD_BUF;
const uint32_t write_at_cmd_buf __attribute__((used)) = NOC_MODE == DEDICATED_NOC_PER_DM ? BRISC_AT_CMD_BUF : MULTI_NOC_BRISC_AT_CMD_BUF;

void kernel_launch() {

#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    firmware_kernel_common_init((void tt_l1_ptr *)(__kernel_init_local_l1_base));

    if constexpr (NOC_MODE == DEDICATED_NOC_PER_DM) {
        noc_local_state_init(noc_index);
    } else {
        noc_local_state_init(NOC_0);
        noc_local_state_init(NOC_1);
    }

    {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        kernel_main();
    }
#endif
}
