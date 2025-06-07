// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "compute_kernel_api.h"
#include <sfpi.h>

// Check initialized and uninitialized data are initialized correctly.

using namespace sfpi;
namespace NAMESPACE {
volatile uint32_t global __attribute__((used)) = 0x12345678;
volatile uint32_t zero __attribute__((used));

// Do not let compiler propagate knowledge of the initialized value;
static uint32_t __attribute__((noinline)) get(volatile uint32_t *ptr) {
    return *ptr;
}

void MAIN {
#if COMPILE_FOR_TRISC == 1  // compute
#include "pre.inc"
    {
        vUInt g = get(&global);
        FAIL_IF(g != 0x12345678);
    }
    {
        vUInt z = get(&zero);
        FAIL_IF(z != 0);
    }
#include "post.inc"
#endif
}
}  // namespace NAMESPACE
