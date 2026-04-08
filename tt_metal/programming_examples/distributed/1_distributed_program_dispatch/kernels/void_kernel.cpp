// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"  // required in all kernels using DEVICE_PRINT
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // Nothing to compute. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    DEVICE_PRINT_MATH("Hello, World! I'm running a void compute kernel.\n");
}
