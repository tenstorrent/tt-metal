// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TRISC is a compute processor, not a data movement processor.
// The NOC non-blocking API is not available for TRISC.
#if !defined(COMPILE_FOR_TRISC)

// =============================================================================
// Quasar NOC Non-Blocking API
// =============================================================================
//
// Two implementations exist:
//
// V1 (noc_nonblocking_api_v1.h) - Legacy Path
//   Uses memory-mapped NOC register writes (NOC_CMD_BUF_WRITE_REG / NOC_CMD_BUF_READ_REG),
//   same mechanism as Wormhole and Blackhole.
//   Kept for compatibility and bringup; not recommended for production as it is super slow.
//
// V2 (noc_nonblocking_api_v2.h) - Custom Instruction Path (Default)
//   Uses RISC-V custom instructions (RoCC) to program the Quasar overlay's command buffers
//   directly. Register writes go through the ROCC interface.
//
//   Each data-movement core has 3 overlay command buffers:
//     - Command Buffer 0 (OVERLAY_WR_CMD_BUF): Complex buffer, typically used for writes.
//     - Command Buffer 1 (OVERLAY_RD_CMD_BUF): Complex buffer, typically used for reads.
//     - Simple Command Buffer (OVERLAY_AT_CMD_BUF, index 2): Used for atomics and inline
//       writes. Accessed via a separate instruction set (scmdbuf), does not take a cmd_buf
//       index parameter.
//
//   The two complex buffers (0, 1) are functionally identical and support reads, writes,
//   and DMA transfers. They are programmed via __builtin_riscv_ttrocc_cmdbuf_wr_reg which
//   takes an explicit buffer index (0 or 1). Callers can select either buffer to enable
//   concurrent transaction setup (e.g. fast dispatch uses buffer 0 for bulk writes and
//   buffer 1 for a second independent write stream).
//
//   The simple command buffer is programmed via __builtin_riscv_ttrocc_scmdbuf_wr_reg and
//   SCMDBUF_ISSUE_INLINE_TRANS / SCMDBUF_ISSUE_TRANS. Atomic increments and inline (dword)
//   writes always use this buffer regardless of the cmd_buf argument passed by the caller.
//
// =============================================================================

#define NOC_API_V1

#if !defined(NOC_API_V2)
#include "noc_nonblocking_api_v1.h"
#else
#include "noc_nonblocking_api_v2.h"

// Legacy per-processor cmd_buf aliases for dataflow_cmd_bufs.h compatibility.
// Complex buffers 0,1 are addressed by index; simple buffer (2) uses scmdbuf instructions.
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
