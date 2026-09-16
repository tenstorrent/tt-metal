// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Contains utility functions to perform IO operations of variable length
// over PCIe.

// This implementation is currently not optimized to minimize RISC cycles.
// APIs can be made more stateful, especially for the HostIO op, since the PCIe
// NOC encoding is constant.

// Both helpers program the MID register from the full 64 bit PCIe address, and the plain
// noc_async_read/write paths no longer rewrite it. Each call refreshes it, so a run of calls needs no
// cleanup between them, but the caller must call noc_async_read_clear_pcie_state or
// noc_async_write_clear_pcie_state on the matching command buffer once the run is done. Otherwise the
// next ordinary on-chip transaction on that buffer, in this kernel or the next one on this core, is
// silently routed to host memory.

FORCE_INLINE void noc_async_wide_write_any_len_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t dst_noc_addr, uint64_t dst_addr, uint32_t len_bytes) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            noc, src_addr, dst_noc_addr, dst_addr, NOC_MAX_BURST_SIZE, 1);
        len_bytes -= NOC_MAX_BURST_SIZE;
        src_addr += NOC_MAX_BURST_SIZE;
        dst_addr += NOC_MAX_BURST_SIZE;
    }
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        noc, src_addr, dst_noc_addr, dst_addr, len_bytes, 1);
}

FORCE_INLINE void noc_async_wide_read_any_len_with_state(
    uint32_t noc, uint64_t src_noc_encoding, uint64_t src_addr, uint32_t dst_addr, uint32_t len_bytes) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            noc, src_noc_encoding, src_addr, dst_addr, NOC_MAX_BURST_SIZE);
        len_bytes -= NOC_MAX_BURST_SIZE;
        src_addr += NOC_MAX_BURST_SIZE;
        dst_addr += NOC_MAX_BURST_SIZE;
    }
    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
        noc, src_noc_encoding, src_addr, dst_addr, len_bytes);
}
