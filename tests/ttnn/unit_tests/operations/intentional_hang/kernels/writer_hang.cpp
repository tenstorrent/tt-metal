// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for intentional_hang. Blocks forever on cb_wait_front since
// the compute kernel never pushes any tile into cb_out. This produces
// a deterministic device-side hang that surfaces to the host as a
// dispatch/operation timeout from system_memory_manager.cpp.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    cb_wait_front(cb_out, 1);  // deadlock — compute never pushes
}
