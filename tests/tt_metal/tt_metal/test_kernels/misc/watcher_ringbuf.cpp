// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/ring_buffer.h"

/*
 * A test for the watcher waypointing feature.
*/
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif

    // Conditionally enable using defines for each trisc
#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or \
    (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or \
    (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
    for (uint32_t idx = 0; idx < 40; idx++) {
        WATCHER_RING_BUFFER_PUSH((idx + 1) + (idx << 16));
    }
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
}
#else
}
}
#endif
