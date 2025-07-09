// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    // Simply print a message to show that the Data Movement kernel is running.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.
    DPRINT << "My logical coordinates are " << (uint32_t)get_absolute_logical_x() << ","
           << (uint32_t)get_absolute_logical_y() << ENDL();
    // The DPRINT_DATA0 and DPRINT_DATA1 macros are used to print messages selectively on Data Movement cores 0 and 1
    // respectively. Otherwise by default DPRINT will print on all cores.
    DPRINT_DATA0(DPRINT << "Hello, host, I am running a void data movement kernel on Data Movement core 0." << ENDL());
    DPRINT_DATA1(DPRINT << "Hello, host, I am running a void data movement kernel on Data Movement core 1." << ENDL());
}
