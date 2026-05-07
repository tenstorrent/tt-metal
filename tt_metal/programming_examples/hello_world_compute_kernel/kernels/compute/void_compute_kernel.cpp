// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"  // required in all kernels using DEVICE_PRINT
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // Nothing to compute. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    DEVICE_PRINT_MATH("Hello, I am the MATH core running the compute kernel.\n");
    DEVICE_PRINT_UNPACK("Hello, I am the UNPACK core running the compute kernel.\n");
    DEVICE_PRINT_PACK("Hello, I am the PACK core running the compute kernel.\n");
}
