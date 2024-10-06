// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NULL kernel is not 0, subtract off overhead
#if KERNEL_BYTES > 16
constexpr uint32_t empty_kernel_bytes = 16;
uint8_t data1[KERNEL_BYTES - empty_kernel_bytes] __attribute__ ((section ("l1_data_test_only"))) __attribute__((used));
#endif

#ifdef KERNEL_GLOBAL
volatile uint32_t global = 4;
#endif

#ifdef COMPILE_FOR_TRISC
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
}
}
#else
void kernel_main() {
}
#endif
