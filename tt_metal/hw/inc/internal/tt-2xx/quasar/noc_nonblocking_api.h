// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TRISC is a compute processor, not a data movement processor.
// The NOC non-blocking API is not available for TRISC.
#if !defined(COMPILE_FOR_TRISC)

// NOC Non-blocking API version selector
// Define NOC_API_V2 to use the V2 implementation, otherwise V1 is used by default

#define NOC_API_V2

#if !defined(NOC_API_V2)
#include "noc_nonblocking_api_v1.h"
#else
#include "noc_nonblocking_api_v2.h"

// Map legacy per-processor cmd buf names to the overlay buffers so
// dataflow_cmd_bufs.h works without modification.
// Quasar has 2 normal cmd buffers (0,1) + 1 simple buffer (2, for atomics & inline writes).
// cmd_buf==0,1 use normal cmdbuf instructions; cmd_buf==2 routes to simple buffer (scmdbuf).
constexpr uint32_t DYNAMIC_NOC_NCRISC_WR_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_NCRISC_WR_REG_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_NCRISC_AT_CMD_BUF = OVERLAY_AT_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_NCRISC_RD_CMD_BUF = OVERLAY_RD_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_BRISC_WR_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_BRISC_AT_CMD_BUF = OVERLAY_AT_CMD_BUF;
constexpr uint32_t DYNAMIC_NOC_BRISC_RD_CMD_BUF = OVERLAY_RD_CMD_BUF;

constexpr uint32_t NCRISC_WR_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t NCRISC_RD_CMD_BUF = OVERLAY_RD_CMD_BUF;
constexpr uint32_t NCRISC_WR_REG_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t NCRISC_AT_CMD_BUF = OVERLAY_AT_CMD_BUF;

constexpr uint32_t BRISC_WR_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t BRISC_RD_CMD_BUF = OVERLAY_RD_CMD_BUF;
constexpr uint32_t BRISC_WR_REG_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t BRISC_AT_CMD_BUF = OVERLAY_AT_CMD_BUF;

#endif  // NOC_API_V2

#endif  // !defined(COMPILE_FOR_TRISC)
