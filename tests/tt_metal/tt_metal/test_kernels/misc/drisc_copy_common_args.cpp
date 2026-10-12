// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t l1_address = get_compile_time_arg_val(0);
    constexpr uint32_t num_args = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<tt_l1_ptr uint32_t*>(l1_address);
    for (uint32_t i = 0; i < num_args; i++) {
        result[i] = get_common_arg_val<uint32_t>(i);
    }
}
