// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/ring_buffer.h"
#include "api/compile_time_args.h"

/*
 * A test for the watcher ringbuffer feature.
*/
#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#endif
#ifdef MPSC_RING_BUFFER
#include "internal/hw_thread.h"
#endif

// Single-TRISC filter: match UCK binary type to target TRISCx define
#if defined(COMPILE_FOR_TRISC) && !defined(MULTI_THREADED_TEST)
#if (defined(UCK_CHLKC_UNPACK) && defined(TRISC0)) || \
    (defined(UCK_CHLKC_MATH) && defined(TRISC1)) || \
    (defined(UCK_CHLKC_PACK) && defined(TRISC2)) || \
    (defined(UCK_CHLKC_ISOLATE_SFPU) && defined(TRISC3))
#define IS_TARGET_TRISC
#endif
#endif

void kernel_main() {
    constexpr uint32_t num_pushes = get_compile_time_arg_val(0);

#ifdef MPSC_RING_BUFFER
    // MPSC: push (thread_idx << 16) | seq
    uint32_t thread_idx = internal_::get_hw_thread_idx();

#if defined(COMPILE_FOR_DM) && !defined(MULTI_THREADED_TEST)
    constexpr uint32_t target_dm = get_compile_time_arg_val(1);
    if (target_dm != thread_idx) return;
#endif

#if !defined(COMPILE_FOR_TRISC) || defined(MULTI_THREADED_TEST) || defined(IS_TARGET_TRISC)
    for (uint32_t seq = 0; seq < num_pushes; seq++) {
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | seq);
    }
#endif

#else  // SPSC: push (idx << 16) | (idx + 1)
#if !defined(COMPILE_FOR_TRISC) || defined(IS_TARGET_TRISC)
    for (uint32_t idx = 0; idx < num_pushes; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx << 16) | (idx + 1));
    }
#endif
#endif
}
