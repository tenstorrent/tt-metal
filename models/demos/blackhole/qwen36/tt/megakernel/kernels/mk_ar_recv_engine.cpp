// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR: the engine form of AR_RECV as a standalone kernel (ar_phase.py ArPhase test path). All config comes
// from defines (MKAR_*, as the engine's owner DM gets them) and all args from the runtime list (MKAR_ALL_RT=1):
// [common block (MKAR_NCOMMON words), role block].
#include <cstdint>
#include "mk_ar_recv.hpp"

void kernel_main() {
    static mkar::Recv r;
    if (!r.init(0, MKAR_NCOMMON)) {
        return;
    }
    for (uint32_t it = 0; it < MKAR_ITERS; ++it) {
        r.iter(it);
    }
    r.fini();
}
