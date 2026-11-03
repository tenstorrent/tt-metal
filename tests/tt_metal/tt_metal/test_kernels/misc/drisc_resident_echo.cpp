// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRISC test kernel for the resident DRAM-core kernel API (experimental/dram_core_kernels.hpp): stays resident until
// the host writes a nonzero stop word, echoing every request the host posts in the meantime.
// L1 words at the runtime-arg base: [0] stop (host), [1] ready (kernel: kReadyMagic while running, 0 on exit),
// [2] request count (host), [3] requests served (kernel), [4] last request echoed (kernel: request + seed).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t kReadyMagic = 0xD815C0DE;
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t seed = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    words[3] = 0;
    words[4] = 0;
    words[1] = kReadyMagic;
    uint32_t served = 0;
    while (words[0] == 0) {
        invalidate_l1_cache();
        const uint32_t posted = words[2];
        if (posted != served) {
            served = posted;
            words[4] = posted + seed;
            words[3] = served;
        }
    }
    words[1] = 0;
}
