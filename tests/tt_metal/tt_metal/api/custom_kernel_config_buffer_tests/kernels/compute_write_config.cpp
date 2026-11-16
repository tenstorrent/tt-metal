// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/compute/common.h"

// Each TRISC writes back what it read from its kernel-config block, starting at result word `slot`:
// UNPACK, MATH and PACK each write their runtime arg plus 0, 1 and 2, then UNPACK writes a common
// runtime arg.
//
// Compile-time args: [slot]. Common runtime args: [result address, value]. Runtime args: [value].
void kernel_main() {
    constexpr uint32_t slot = get_compile_time_arg_val(0);
    tt_l1_ptr uint32_t* result = reinterpret_cast<tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(0)) + slot;
    const uint32_t value = get_arg_val<uint32_t>(0);
    UNPACK(result[0] = value);
    MATH(result[1] = value + 1);
    PACK(result[2] = value + 2);
    UNPACK(result[3] = get_common_arg_val<uint32_t>(1));
}
