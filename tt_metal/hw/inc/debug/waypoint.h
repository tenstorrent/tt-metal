// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/waypoint.h
//
// This file implements a method to log "waypoints" in device code
// Each riscv processor has a 4 byte debug status mailbox in L1
// Use the macro WATCHER_WAYPOINT(...) to log up to 4 characters in the mailbox
// The host watcher thread prints these periodically
// All functionaly gated behind defined WATCHER_ENABLED
//
#pragma once

#include <utility>

#include "dev_msgs.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_STATUS)
#include <cstddef>

template <size_t N, size_t... Is>
constexpr uint32_t fold(const char (&s)[N], std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) <= 4, "Up to 4 characters allowed in WATCHER_WAYPOINT");
    return ((static_cast<uint32_t>(s[Is]) << (8 * Is)) | ...);
}

template <size_t N>
constexpr uint32_t helper(const char (&s)[N]) {
    return fold(s, std::make_index_sequence<N - 1>{});
}

template<uint32_t x>
inline void write_debug_waypoint(volatile tt_l1_ptr uint32_t *debug_waypoint) {
    *debug_waypoint = x;
}

#if defined(COMPILE_FOR_BRISC)
#define WATCHER_WAYPOINT_MAILBOX_OFFSET 0
#elif defined(COMPILE_FOR_NCRISC)
#define WATCHER_WAYPOINT_MAILBOX_OFFSET 1
#elif defined(COMPILE_FOR_ERISC)
#define WATCHER_WAYPOINT_MAILBOX_OFFSET 0
#elif defined(COMPILE_FOR_IDLE_ERISC)
#define WATCHER_WAYPOINT_MAILBOX_OFFSET 0
#else
#define WATCHER_WAYPOINT_MAILBOX_OFFSET (2 + COMPILE_FOR_TRISC)
#endif

#define WATCHER_WAYPOINT_MAILBOX \
    (volatile tt_l1_ptr uint32_t *)&((*GET_MAILBOX_ADDRESS_DEV(watcher.debug_waypoint))[WATCHER_WAYPOINT_MAILBOX_OFFSET])

#define WAYPOINT(x) write_debug_waypoint<helper(x)>(WATCHER_WAYPOINT_MAILBOX)

#else  // !WATCHER_ENABLED

#define WAYPOINT(x)

#endif  // WATCHER_ENABLED
