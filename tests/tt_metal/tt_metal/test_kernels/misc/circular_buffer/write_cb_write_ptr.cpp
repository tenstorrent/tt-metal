// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Writes CB 0's write pointer, from its CB config, to word 0 of the result page.
//
// Common runtime args: [result address].
void kernel_main() {
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(0));
    result[0] = get_write_ptr(0);
}
