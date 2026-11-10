// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR: the engine form of the owner compute phase as a standalone kernel (ar_phase.py ArPhase test path);
// config by defines (MKAR_*), args [has_work, ring, chunk tiles].
#include <cstdint>
#include "mk_ar_reduce.hpp"

void kernel_main() {
    mkar::Reduce red;
    if (!red.init(0)) {
        return;
    }
    compute_kernel_hw_startup(MKAR_CB_RED, MKAR_CB_RED, MKAR_CB_OUT);
    for (uint32_t it = 0; it < MKAR_ITERS; ++it) {
        red.iter();
    }
}
