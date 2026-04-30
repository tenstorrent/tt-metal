// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"  // required in all kernels using DEVICE_PRINT
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // Nothing to compute. Print response message.
    DEVICE_PRINT_MATH("Void compute kernel is running.\n\n");

    // The user is encouraged to play around with tile compute behavior (i.e., element-wise math, column-wise or
    // row-wise transformations, etc.) as an exercise.
}
