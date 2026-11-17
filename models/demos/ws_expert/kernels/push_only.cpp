// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bench helper (data movement): pushes CT-listed CB pages without data (compute timing benches whose results are not
// checked). CT: 0 count, then (cb, pages) pairs.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = kernel_compile_time_args[1 + 2 * i], pages = kernel_compile_time_args[2 + 2 * i];
        cb_reserve_back(c, pages);
        cb_push_back(c, pages);
    }
}
