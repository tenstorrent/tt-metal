// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"

// Producer-side helpers of the software-counter credits (FABRIC_STREAM_REGISTERS_AVAILABLE 0). An architecture that
// reports these failures, or reaches the consumer, differently supplies its own version of this header on its include
// path; callers include it by this root-relative path.

// Software credits: write a producer's free-running write count @p write_count into the SenderChannelProducerCursor
// write_counter at L1 address @p write_counter_addr on core (@p noc_x, @p noc_y), once every non-posted write this core
// issued on @p noc has departed (noc_async_writes_flushed()). Posted writes are not covered.
FORCE_INLINE void fabric_publish_producer_write_count(
    uint8_t noc_x, uint8_t noc_y, uint32_t write_counter_addr, uint32_t write_count, uint8_t noc = noc_index) {
    noc_async_writes_flushed(noc);
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(noc_x, noc_y, write_counter_addr, noc), write_count, 0xF, noc);
}

// A producer connecting to a router found @p router_credit_mode (EDMChannelWorkerLocationInfo::credit_mode) differ from
// its own: the router would poll a credit nobody writes. Never returns; the waypoint names the hang for the watcher.
[[noreturn]] inline void fabric_credit_mode_mismatch(uint32_t /*router_credit_mode*/) {
    WAYPOINT("FCMM");
    while (true) {
    }
}
