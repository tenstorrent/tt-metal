// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "compute_kernel_api.h"
#include <sfpi.h>

using namespace sfpi;
namespace NAMESPACE {
void MAIN {
#if COMPILE_FOR_TRISC == 1  // compute
#include "pre.inc"
    FAIL_IF(vInt(1) != vInt(1));

    FAIL_IF(vInt(2) != vInt(1));  // This one, line 16

    FAIL_IF(vInt(3) != vInt(1));  // not this one
#include "post.inc"
#endif
}
}  // namespace NAMESPACE
