// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"

namespace NAMESPACE {

    void MAIN {

        // Nothing to compute. Print response message.
        DPRINT_MATH(DPRINT << "Void compute kernel is running." << ENDL() << ENDL());

        // The user is encouraged to play around with tile compute behavior (i.e., element-wise math, column-wise or row-wise transformations, etc.) as an exercise.

    }
}
