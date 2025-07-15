// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"

namespace NAMESPACE {

void MAIN {
    // Nothing to compute. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    DPRINT_MATH(DPRINT << "Hello, I am the MATH core running the compute kernel" << ENDL());
    DPRINT_UNPACK(DPRINT << "Hello, I am the UNPACK core running the compute kernel" << ENDL());
    DPRINT_PACK(DPRINT << "Hello, I am the PACK core running the compute kernel" << ENDL());
}

}  // namespace NAMESPACE
