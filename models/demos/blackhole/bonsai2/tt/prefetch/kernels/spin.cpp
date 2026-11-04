// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Test helper: busy-wait a fixed number of loop iterations (stands in for a non-weight decode op).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t n = get_common_arg_val<uint32_t>(0);
    for (volatile uint32_t i = 0; i < n; ++i) {
    }
}
