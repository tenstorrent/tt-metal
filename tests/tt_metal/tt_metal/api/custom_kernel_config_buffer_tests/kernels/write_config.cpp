// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Writes back what it read from its kernel-config block, starting at result word `slot`: its runtime
// arg, a common runtime arg, and CB 0's write pointer (from the CB config).
//
// Compile-time args: [slot]. Common runtime args: [result address, value]. Runtime args: [value, padding...].
void kernel_main() {
    constexpr uint32_t slot = get_compile_time_arg_val(0);
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(0)) + slot;
    result[0] = get_arg_val<uint32_t>(0);
    result[1] = get_common_arg_val<uint32_t>(1);
    result[2] = get_write_ptr(0);
}
