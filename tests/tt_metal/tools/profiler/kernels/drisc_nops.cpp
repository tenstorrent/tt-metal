// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The DRISC firmware defines no circular buffer interface and dataflow_api.h references it.
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i++) {
        asm("nop");
    }
}
