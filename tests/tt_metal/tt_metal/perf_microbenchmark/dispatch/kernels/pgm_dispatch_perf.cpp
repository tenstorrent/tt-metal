// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NULL kernel is not 0 length so this is smaller than you might think
#if KERNEL_BYTES >= 256
uint64_t data1[30] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 512
uint64_t data2[32] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 1024
uint64_t data3[64] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 2 * 1024
uint64_t data4[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 3 * 1024
uint64_t data5[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 4 * 1024
uint64_t data6[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 5 * 1024
uint64_t data7[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 6 * 1024
uint64_t data8[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 7 * 1024
uint64_t data9[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 8 * 1024
uint64_t data10[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 9 * 1024
uint64_t data11[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 10 * 1024
uint64_t data12[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 11 * 1024
uint64_t data13[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 12 * 1024
uint64_t data14[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 13 * 1024
uint64_t data15[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 14 * 1024
uint64_t data16[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 15 * 1024
uint64_t data17[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
#endif

#if KERNEL_BYTES >= 16 * 1024
uint64_t data18[128] __attribute__ ((section ("l1_data"))) __attribute__((used));
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
