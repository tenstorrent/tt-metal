// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/debug/waypoint.h"

// Producer-side helpers of the software-counter credits (FABRIC_STREAM_REGISTERS_AVAILABLE 0). An architecture that
// reports these failures, or reaches the consumer, differently supplies its own version of this header on its include
// path; callers include it by this root-relative path.

// A producer connecting to a router found @p router_credit_mode (EDMChannelWorkerLocationInfo::credit_mode) differ from
// its own: the router would poll a credit nobody writes. Never returns; the waypoint names the hang for the watcher.
[[noreturn]] inline void fabric_credit_mode_mismatch(uint32_t /*router_credit_mode*/) {
    WAYPOINT("FCMM");
    while (true) {
    }
}
