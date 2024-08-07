// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"

namespace NAMESPACE {

    void MAIN {

        // Nothing to compute. Print respond message.
        DPRINT_MATH(DPRINT << "Void compute kernel is running." << ENDL() << ENDL());

    }

}
