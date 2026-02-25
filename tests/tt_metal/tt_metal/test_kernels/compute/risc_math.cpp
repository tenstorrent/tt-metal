// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "internal/hw_thread.h"

void kernel_main() {
    [[maybe_unused]] int thread_idx = internal_::get_hw_thread_idx();
#ifdef TRISC_PACK
    int32_t A = 1;
    int32_t B = 2;

    DPRINT << "TEST packer" << ENDL();
    DPRINT << A + B + thread_idx << ENDL();
#endif

#ifdef TRISC_UNPACK
    int32_t A = 2;
    int32_t B = 2;

    DPRINT << "TEST unpacker" << ENDL();
    DPRINT << A + B + thread_idx << ENDL();
#endif

#ifdef TRISC_MATH
    int32_t A = 3;
    int32_t B = 2;

    DPRINT << "TEST math" << ENDL();
    DPRINT << A + B + thread_idx << ENDL();
#endif
}
