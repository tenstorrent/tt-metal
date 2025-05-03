// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"

namespace NAMESPACE {

void MAIN {
    // Nothing to compute. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    DPRINT_MATH(DPRINT << "Hello, World! I'm running a void compute kernel." << ENDL());
}

}  // namespace NAMESPACE
