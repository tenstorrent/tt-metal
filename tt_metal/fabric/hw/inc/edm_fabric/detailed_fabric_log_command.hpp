// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>  // offsetof
#include <cstdint>

#include "api/dataflow/dataflow_api.h"      // noc_inline_dw_write, get_noc_addr, InlineWriteDst
#include "api/debug/assert.h"               // ASSERT
#include "hostdev/fabric_telemetry_msgs.h"  // FabricTelemetry (mailbox layout)
#include "dev_mem_map.h"                    // MEM_AERISC_FABRIC_TELEMETRY_BASE

// [debug] Worker-side control API for the fabric routers' detailed flow-control logging ([rxlog]/[txlog]).
//
// A worker kernel calls start_detailed_logging() / stop_detailed_logging() on each fabric eth router it drives
// to open/close that router's trace window. This wraps the single L1 control word the router polls, so callers
// never hand-compute the mailbox address or issue a raw NOC write into fabric L1.
//
// The control word is FabricTelemetry::scratch[0] in the router's telemetry mailbox (at
// MEM_AERISC_FABRIC_TELEMETRY_BASE). The router (fabric_erisc_router.cpp main loop) reads that SAME slot each
// pass: non-zero => window open (the value is stamped into the trace as its correlation/batch id), 0 => closed.
// Because both sides go through FabricTelemetry::scratch, the command and the router's interpretation stay in
// sync by construction.

namespace tt::tt_fabric {

// L1 byte address, within a fabric eth router's telemetry mailbox, of the detailed-logging control word.
FORCE_INLINE uint32_t detailed_logging_control_l1_addr() {
    return static_cast<uint32_t>(MEM_AERISC_FABRIC_TELEMETRY_BASE + offsetof(FabricTelemetry, scratch));
}

// Open the detailed flow-control logging window on the fabric eth router at (edm_noc_x, edm_noc_y).
// window_id is a NONZERO correlation/batch id stamped into the trace (0 is reserved for "stopped"); it lets a
// dumped trace file be tied back to the producer/window that generated it. Single inline dword NOC write; the
// router reads it locally on its next main-loop pass.
FORCE_INLINE void start_detailed_logging(uint32_t edm_noc_x, uint32_t edm_noc_y, uint32_t window_id) {
    ASSERT(window_id != 0);  // 0 means "logging stopped" -- pass a nonzero batch id
    noc_inline_dw_write<InlineWriteDst::L1>(
        get_noc_addr(edm_noc_x, edm_noc_y, detailed_logging_control_l1_addr()), window_id);
}

// Close the detailed logging window on the router at (edm_noc_x, edm_noc_y). On seeing the 0, the router
// finalizes its trace (flushes the L1 tail to DRAM) so the host reader recovers the complete record array.
FORCE_INLINE void stop_detailed_logging(uint32_t edm_noc_x, uint32_t edm_noc_y) {
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(edm_noc_x, edm_noc_y, detailed_logging_control_l1_addr()), 0);
}

}  // namespace tt::tt_fabric
