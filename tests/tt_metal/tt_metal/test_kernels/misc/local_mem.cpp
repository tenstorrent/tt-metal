// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/assert.h"

volatile uint64_t global_eight_byte_val = 0xABCD0123ABCD0123;

volatile uint32_t nonzero[4] = {
    0xAABB0000,
    0xAABB0001,
    0xAABB0002,
    0xAABB0003,
};
volatile uint32_t zero[4] = {0, 0, 0, 0};

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif

    if (nonzero[0] != 0xAABB0000 || nonzero[1] != 0xAABB0001 || nonzero[2] != 0xAABB0002 || nonzero[3] != 0xAABB0003) {
        ASSERT(0);
        while (1);
    }

    if (zero[0] != 0 || zero[1] != 0 || zero[2] != 0 || zero[3] != 0) {
        ASSERT(0);
        while (1);
    }

    // Ensure back to back runs get fresh data
    nonzero[0] = 0xdeadbeef;
    nonzero[1] = 0xdeadbeef;
    nonzero[2] = 0xdeadbeef;
    nonzero[3] = 0xdeadbeef;

    zero[0] = 0xdeadbeef;
    zero[1] = 0xdeadbeef;
    zero[2] = 0xdeadbeef;
    zero[3] = 0xdeadbeef;

    if (global_eight_byte_val != 0xABCD0123ABCD0123) {
        ASSERT(0);
        while (1);
    }
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)
}
#else
}
}
#endif
