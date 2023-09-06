/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//
// debug_status.h
//
// This file implements a method to log "waypoints" in device code
// Each riscv processor has a 4 byte debug status mailbox in L1
// Use the macro DEBUG_STATUS(...) to log up to 4 characters in the mailbox
// The host watcher thread prints these periodically
// All functionaly gated behind defined WATCHER_ENABLED
//
#pragma once

#include "dev_mem_map.h"

#if defined (WATCHER_ENABLED)

inline uint32_t get_debug_status()
{
    return 0;
}

template<typename... Types>
inline uint32_t get_debug_status(uint32_t c, Types... rest)
{
    return (c == 0) ? 0 : (get_debug_status(rest...) << 8) | c;
}

template<typename... Types>
inline void write_debug_status(volatile tt_l1_ptr uint32_t *debug_status, uint32_t c, Types... rest)
{
    uint32_t x = get_debug_status(c, rest...);
    *debug_status = x;
}

#if defined(COMPILE_FOR_BRISC)
#define DEBUG_STATUS_MAILBOX (volatile tt_l1_ptr uint32_t *)MEM_DEBUG_BRISC_STATUS_MAILBOX_ADDRESS
#elif defined(COMPILE_FOR_NCRISC)
#define DEBUG_STATUS_MAILBOX (volatile tt_l1_ptr uint32_t *)MEM_DEBUG_NCRISC_STATUS_MAILBOX_ADDRESS
#else
#define GET_TRISC_DEBUG_STATUS_MAILBOX_EVAL(x, t, y) x##t##y
#define GET_TRISC_DEBUG_STATUS_MAILBOX(x, t, y) GET_TRISC_DEBUG_STATUS_MAILBOX_EVAL(x, t, y)
#define DEBUG_STATUS_MAILBOX (volatile tt_l1_ptr uint32_t *)GET_TRISC_DEBUG_STATUS_MAILBOX(MEM_DEBUG_TRISC, COMPILE_FOR_TRISC, _STATUS_MAILBOX_ADDRESS)
#endif

#define DEBUG_STATUS(x...) write_debug_status(DEBUG_STATUS_MAILBOX, x, 0)

#else // !WATCHER_ENABLED

#define DEBUG_STATUS(x...)

#endif // WATCHER_ENABLED
