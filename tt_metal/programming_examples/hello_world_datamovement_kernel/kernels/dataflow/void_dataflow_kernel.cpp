// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"  // required in all kernels using DEVICE_PRINT
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Simply print a message to show that the Data Movement kernel is running.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.
    DEVICE_PRINT("My logical coordinates are {},{}\n", get_absolute_logical_x(), get_absolute_logical_y());
    // The DEVICE_PRINT_DATA0 and DEVICE_PRINT_DATA1 macros are used to print messages selectively on Data Movement
    // cores 0 and 1 respectively. Otherwise by default DEVICE_PRINT will print on all cores.
    DEVICE_PRINT_DATA0("Hello, host, I am running a void data movement kernel on Data Movement core 0.\n");
    DEVICE_PRINT_DATA1("Hello, host, I am running a void data movement kernel on Data Movement core 1.\n");
}
