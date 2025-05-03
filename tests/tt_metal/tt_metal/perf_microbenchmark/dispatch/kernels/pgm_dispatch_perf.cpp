// SPDX-FileCopyrightText: © 2023, 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// An empty kernel is 16 bytes, so only pad above that to fake a
// bigger kernel.
#define EMPTY_KERNEL_BYTES 16
#if KERNEL_BYTES > EMPTY_KERNEL_BYTES
[[gnu::section(".text"), gnu::used]]
static uint8_t lorem_ipsum[KERNEL_BYTES - EMPTY_KERNEL_BYTES];
#endif

#ifdef KERNEL_GLOBAL
[[gnu::section(".data"), gnu::used]]
static uint32_t global;
#endif

#ifdef COMPILE_FOR_TRISC
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {}
}  // namespace NAMESPACE
#else
void kernel_main() {}
#endif
