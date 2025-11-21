// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// An empty kernel is 16 bytes, so only pad above that to fake a bigger kernel.
#define EMPTY_KERNEL_BYTES 16
#if KERNEL_BYTES > EMPTY_KERNEL_BYTES
[[gnu::section(".text"), gnu::used]]
static uint8_t lorem_ipsum[KERNEL_BYTES - EMPTY_KERNEL_BYTES];
#endif

#ifdef COMPILE_FOR_TRISC
// Compute kernel path
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#if KERNEL_BYTES > EMPTY_KERNEL_BYTES
    // Touch the padding to prevent optimization
    lorem_ipsum[0] = 1;
#endif
}
}  // namespace NAMESPACE
#else
// Dataflow kernel path
#include "dataflow_api.h"
void kernel_main() {
#if KERNEL_BYTES > EMPTY_KERNEL_BYTES
    // Touch the padding to prevent optimization
    lorem_ipsum[0] = 1;
#endif
}
#endif
