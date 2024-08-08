// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NULL kernel is not 0, subtract off overhead
#if KERNEL_BYTES > 30
uint8_t data1[KERNEL_BYTES-30] __attribute__ ((section ("l1_data"))) __attribute__((used));
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
