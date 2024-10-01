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
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"

#include <kernel_includes.hpp>

uint8_t noc_index = NOC_INDEX;
extern uint32_t __kernel_init_local_l1_base[];

constexpr uint32_t read_cmd_buf __attribute__((used)) = NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf __attribute__((used)) = NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf __attribute__((used)) = NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf __attribute__((used)) = NCRISC_AT_CMD_BUF;

void kernel_launch() {
    DeviceZoneScopedMainChildN("ERISC-KERNEL");
    firmware_kernel_common_init((void tt_l1_ptr *)__kernel_init_local_l1_base);

    noc_local_state_init(noc_index);

    kernel_main();
}
