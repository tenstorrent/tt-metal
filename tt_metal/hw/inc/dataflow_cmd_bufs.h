// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"

#if defined(KERNEL_BUILD)
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t read_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_RD_CMD_BUF : DYNAMIC_NOC_BRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_WR_CMD_BUF : DYNAMIC_NOC_BRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf =
    NOC_MODE == DM_DEDICATED_NOC ? BRISC_WR_REG_CMD_BUF : DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? BRISC_AT_CMD_BUF : DYNAMIC_NOC_BRISC_AT_CMD_BUF;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t read_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_RD_CMD_BUF : DYNAMIC_NOC_NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_WR_CMD_BUF : DYNAMIC_NOC_NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf =
    NOC_MODE == DM_DEDICATED_NOC ? NCRISC_WR_REG_CMD_BUF : DYNAMIC_NOC_NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NOC_MODE == DM_DEDICATED_NOC ? NCRISC_AT_CMD_BUF : DYNAMIC_NOC_NCRISC_AT_CMD_BUF;
#else  // use the default cmf buffers for compute/eth
constexpr uint32_t read_cmd_buf = NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf = NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NCRISC_AT_CMD_BUF;
#endif
#else  // FW build
constexpr uint32_t read_cmd_buf = NCRISC_RD_CMD_BUF;
constexpr uint32_t write_cmd_buf = NCRISC_WR_CMD_BUF;
constexpr uint32_t write_reg_cmd_buf = NCRISC_WR_REG_CMD_BUF;
constexpr uint32_t write_at_cmd_buf = NCRISC_AT_CMD_BUF;
#endif
