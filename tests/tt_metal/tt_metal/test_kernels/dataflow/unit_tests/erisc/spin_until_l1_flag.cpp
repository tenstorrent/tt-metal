// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Keeps an ethernet core inside a kernel until the host releases it: writes 1 to the ack address once running,
// then spins until the host writes a non-zero value to the release address.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ack_addr = get_arg_val<uint32_t>(0);
    const uint32_t release_addr = get_arg_val<uint32_t>(1);

    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_addr) = 1;
    auto* release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release_addr);
    while (*release == 0) {
        invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
    }
}
