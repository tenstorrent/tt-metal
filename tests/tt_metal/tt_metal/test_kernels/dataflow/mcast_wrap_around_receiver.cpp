// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver kernel for wrap-around multicast tests.
// Passive receiver - data arrives via NoC multicast from sender kernel.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // No work needed - receiver just needs to be running to accept multicast data
}
