// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
//   Kept for compatibility and bringup; not recommended for production as it is slow and prone to race conditions.
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
//   concurrent transaction setup. Fast dispatch keeps every write on buffer 0 (NCRISC_WR_CMD_BUF and
//   NCRISC_WR_REG_CMD_BUF both alias it, so the stateful inline writes land there too) and every read on
//   buffer 1; dispatch_s passes cmd_buf 1 only to noc_fast_write_dw_inline, which uses the simple buffer.
//   So a buffer armed for stateful reads never carries a write.
//
//   The simple command buffer is programmed via __builtin_riscv_ttrocc_scmdbuf_wr_reg and
//   SCMDBUF_ISSUE_INLINE_TRANS / SCMDBUF_ISSUE_TRANS. Atomic increments and inline (dword)
//   writes always use this buffer regardless of the cmd_buf argument passed by the caller.
//
// =============================================================================

#if defined(NOC_API_V3)
#include "noc_nonblocking_api_v3.h"
#elif defined(NOC_API_V1)
#include "noc_nonblocking_api_v1.h"
#else
#include "noc_nonblocking_api_v2.h"
#endif

#include "tensix_neo_reg.h"

// Quasar-only: the per-VC initiator throttler (NOC_CFG 0x2C..0x30) desynchronises the router state machine for that
// VC, so it stays at reset (0).
constexpr uint32_t NOC_THROTTLER_RESET_VALUE = 0;
static_assert(
    VC_THROTTLER_CYCLES_PER_WINDOW - VC_THROTTLER_HANDSHAKES_PER_WINDOW_0 == 4,
    "per-VC throttler NOC_CFG ids must be contiguous");

// True when the per-VC initiator throttler of every NOC on this tile reads 0. Reads NIU MMIO: DM0 calls it at boot.
inline bool noc_throttler_at_reset() {
    for (uint32_t noc = 0; noc < NUM_NOCS; ++noc) {
        // The HANDSHAKES registers use all 32 bits.
        for (uint32_t reg_id = VC_THROTTLER_HANDSHAKES_PER_WINDOW_0; reg_id < VC_THROTTLER_CYCLES_PER_WINDOW;
             ++reg_id) {
            if (NOC_CFG_READ_REG(noc, reg_id) != NOC_THROTTLER_RESET_VALUE) {
                return false;
            }
        }
        if ((NOC_CFG_READ_REG(noc, VC_THROTTLER_CYCLES_PER_WINDOW) &
             NOC_CONFIG_VC_THROTTLER_CYCLES_PER_WINDOW_DATA_MASK) != NOC_THROTTLER_RESET_VALUE) {
            return false;
        }
    }
    return true;
}

#if !defined(NOC_API_V1)
// Legacy per-processor cmd_buf aliases for dataflow_cmd_bufs.h compatibility (V1 defines
// its own). Complex buffers 0,1 are addressed by index; simple buffer (2) uses scmdbuf
// instructions.
constexpr uint32_t NCRISC_WR_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t NCRISC_RD_CMD_BUF = OVERLAY_RD_CMD_BUF;
constexpr uint32_t NCRISC_WR_REG_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t NCRISC_AT_CMD_BUF = OVERLAY_AT_CMD_BUF;

constexpr uint32_t BRISC_WR_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t BRISC_RD_CMD_BUF = OVERLAY_RD_CMD_BUF;
constexpr uint32_t BRISC_WR_REG_CMD_BUF = OVERLAY_WR_CMD_BUF;
constexpr uint32_t BRISC_AT_CMD_BUF = OVERLAY_AT_CMD_BUF;
#endif

#endif  // !defined(COMPILE_FOR_TRISC)
