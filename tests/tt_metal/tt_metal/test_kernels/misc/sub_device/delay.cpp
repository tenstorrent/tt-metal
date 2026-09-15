// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Spins for the number of iterations given in the first runtime arg, to keep a core busy for a
// controllable amount of time.
void kernel_main() {
    uint32_t num_iterations = get_arg_val<uint32_t>(0);

    for (volatile uint32_t i = 0; i < num_iterations; i++) {
    }
}
