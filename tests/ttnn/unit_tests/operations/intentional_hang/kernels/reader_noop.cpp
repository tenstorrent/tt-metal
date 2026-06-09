// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// No-op reader for intentional_hang. Pushes zero tiles into cb_input.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // intentionally empty — leaves cb_out empty so writer hangs on cb_wait_front
}
