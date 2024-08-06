// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/status.h
//
// This file implements a method to log "waypoints" in device code
// Each riscv processor has a 4 byte debug status mailbox in L1
// Use the macro DEBUG_STATUS(...) to log up to 4 characters in the mailbox
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
    static_assert(sizeof...(Is) <= 4, "Up to 4 characters allowed in DEBUG_STATUS");
    return ((static_cast<uint32_t>(s[Is]) << (8 * Is)) | ...);
}

template <size_t N>
constexpr uint32_t helper(const char (&s)[N]) {
    return fold(s, std::make_index_sequence<N - 1>{});
}

template<uint32_t x>
inline void write_debug_status(volatile tt_l1_ptr uint32_t *debug_status) {
    *debug_status = x;
}

#if defined(COMPILE_FOR_BRISC)
#define DEBUG_STATUS_MAILBOX_OFFSET 0
#elif defined(COMPILE_FOR_NCRISC)
#define DEBUG_STATUS_MAILBOX_OFFSET 1
#elif defined(COMPILE_FOR_ERISC)
#define DEBUG_STATUS_MAILBOX_OFFSET 0
#elif defined(COMPILE_FOR_IDLE_ERISC)
#define DEBUG_STATUS_MAILBOX_OFFSET 0
#else
#define DEBUG_STATUS_MAILBOX_OFFSET (2 + COMPILE_FOR_TRISC)
#endif

#define DEBUG_STATUS_MAILBOX \
    (volatile tt_l1_ptr uint32_t *)&((*GET_MAILBOX_ADDRESS_DEV(watcher.debug_status))[DEBUG_STATUS_MAILBOX_OFFSET])

#define DEBUG_STATUS(x) write_debug_status<helper(x)>(DEBUG_STATUS_MAILBOX)

#else  // !WATCHER_ENABLED

#define DEBUG_STATUS(x)

#endif  // WATCHER_ENABLED
