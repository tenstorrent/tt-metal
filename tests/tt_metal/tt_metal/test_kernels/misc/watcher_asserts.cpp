// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/assert.h"

/*
 * A test for the assert feature.
*/
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif

    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);

    // Conditionally enable using defines for each trisc
#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or \
    (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or \
    (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC))
    ASSERT(a != b);
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)
}
#else
}
}
#endif
