// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#include "api/compute/common.h"
#endif
#include "api/debug/device_print.h"

/*
 * Test kernel for printing same message from multiple iterations, used to verify that the DEVICE_PRINT locking
 * mechanism prevents interleaved/corrupted output when multiple RISCs print simultaneously.
 */
void kernel_main() {
    uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < count; i++) {
        DEVICE_PRINT("Test iteration: {}\n", i);
    }
}
