// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

// Ensure this is set: export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'
void kernel_main() {

    // Nothing to move. Print response message.
    DPRINT_DATA0(DPRINT << "Void outbound kernel is running." << ENDL()<< ENDL());
    DPRINT_DATA1(DPRINT << "Void outbound kernel is running." << ENDL()<< ENDL());

    // The user is encouraged to play around with tile outbound behavior (i.e., store into DRAM, pipeline to other Tensix cores, etc.) as an exercise.
}
