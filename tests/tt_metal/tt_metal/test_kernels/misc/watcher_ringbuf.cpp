// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/ring_buffer.h"
#include "api/compile_time_args.h"
#if defined(ARCH_QUASAR)
// Use internal_::get_hw_thread_idx() to get HAL processor index (0-7 for DM, 8+ for TRISC).
// This matches the thread_idx embedded in MPSC write_id by ring_buffer.h, ensuring the
// data payload's thread_idx matches the prefix shown in watcher output (e.g., [Neo0TRISC0]).
// Note: get_my_thread_id() returns a different value (processor-local index) and would mismatch.
#include "internal/hw_thread.h"
#endif

/*
 * A test for the watcher ring buffer feature.
*/
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif

void kernel_main() {
    constexpr uint32_t num_pushes = get_compile_time_arg_val(0);

#if defined(COMPILE_FOR_DM)
    uint32_t thread_idx = internal_::get_hw_thread_idx();
#if !defined(MULTI_DM_TEST)
    // Single-DM test: only specified DM runs
    constexpr uint32_t dm_id = get_compile_time_arg_val(1);
    if (dm_id != thread_idx) {
        return;
    }
#endif
    for (uint32_t seq = 0; seq < num_pushes; seq++) {
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | seq);
    }
    return;
#endif

#if (defined(UCK_CHLKC_UNPACK) && defined(TRISC0)) || \
      (defined(UCK_CHLKC_MATH) && defined(TRISC1)) || \
      (defined(UCK_CHLKC_PACK) && defined(TRISC2))
#if defined(ARCH_QUASAR)
    // Quasar: use HAL thread_idx for MPSC verification
    uint32_t thread_idx = internal_::get_hw_thread_idx();
    for (uint32_t seq = 0; seq < num_pushes; seq++) {
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | seq);
    }
#else
    // WH/BH SPSC: use idx pattern, compile-time filter ensures only matching TRISC runs
    for (uint32_t idx = 0; idx < num_pushes; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx + 1) + (idx << 16));
    }
#endif
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || \
    defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    for (uint32_t idx = 0; idx < num_pushes; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx + 1) + (idx << 16));
    }
#endif
}
