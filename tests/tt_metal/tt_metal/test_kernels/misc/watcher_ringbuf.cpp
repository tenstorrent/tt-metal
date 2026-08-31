// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet and DRAM cores only; TENSIX uses watcher_ringbuf_2_0.cpp.

#include <cstdint>
#include "api/debug/ring_buffer.h"
#include "api/compile_time_args.h"
#if defined(DEBUG_RING_BUFFER_MPSC)
// get_hw_thread_idx(), not get_my_thread_id(), must match ring_buffer.h's MPSC write_id encoding.
#include "internal/hw_thread.h"
#endif

void kernel_main() {
    constexpr uint32_t num_pushes = get_compile_time_arg_val(0);

#if defined(DEBUG_RING_BUFFER_MPSC)
    uint32_t thread_idx = internal_::get_hw_thread_idx();
    for (uint32_t seq = 0; seq < num_pushes; seq++) {
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | seq);
    }
#else
    for (uint32_t idx = 0; idx < num_pushes; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx + 1) + (idx << 16));
    }
#endif
}
