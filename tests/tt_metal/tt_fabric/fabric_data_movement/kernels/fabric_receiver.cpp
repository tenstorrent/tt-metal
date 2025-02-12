// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t size = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address + size - sizeof(uint32_t));
    while (*ptr == 0);

    volatile tt_l1_ptr uint32_t* data_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);

    uint32_t l1_addr = get_read_ptr(0);
    volatile tt_l1_ptr uint32_t* cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < size / sizeof(uint32_t); ++i) {
        cb_ptr[i] = data_ptr[i];
        // DPRINT << cb_ptr[i] << ENDL();
    }
    cb_push_back(0, 32);
}
