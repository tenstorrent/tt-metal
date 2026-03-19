// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCIe write smoke test kernel — BRISC (RISCV_0, NOC0)
//
// Minimal isolated test of the noc_wwrite_with_state path.
// Writes one cache line (64 B) of 0xDEADBEEF to a host pinned address,
// waits for the NOC ack, then exits.  The host verifies the value.
//
// Runtime args:
//   0: pcie_xy_enc — PCIe tile NOC XY encoding
//   1: dst_lo      — host pinned address [31:0]
//   2: dst_hi      — host pinned address [63:32]
//
// NOTE: No DPRINT — see persistent_d2h_pusher.cpp for explanation.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

// L1 scratch: 64 bytes of staging data, immediately above the user-space base.
static constexpr uint32_t kScratchBase = 512u * 1024u;  // 0x80000
static constexpr uint32_t kWriteBytes = 64u;            // one cache line
static constexpr uint32_t kWriteWords = kWriteBytes / sizeof(uint32_t);
static constexpr uint32_t kMagicValue = 0xDEADBEEFu;

void kernel_main() {
    const uint32_t pcie_xy_enc = get_arg_val<uint32_t>(0);
    const uint32_t dst_lo = get_arg_val<uint32_t>(1);
    const uint32_t dst_hi = get_arg_val<uint32_t>(2);
    const uint64_t dst_addr = (static_cast<uint64_t>(dst_hi) << 32) | dst_lo;

    // Fill L1 scratch with magic value
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kScratchBase);
    for (uint32_t i = 0; i < kWriteWords; ++i) {
        scratch[i] = kMagicValue;
    }

    // Issue PCIe write: L1 scratch → host pinned memory
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, kScratchBase, pcie_xy_enc, dst_addr, kWriteBytes, 1);

    // Wait for NOC non-posted write ack (device confirmed write reached PCIe tile)
    noc_async_write_barrier();
}
