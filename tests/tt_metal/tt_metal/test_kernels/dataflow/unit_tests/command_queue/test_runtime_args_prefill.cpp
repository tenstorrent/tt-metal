// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    uint32_t l1_write_addr = get_write_ptr(cb_in0);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(l1_write_addr);

    // Write all args to L1 to check
    ptr[0] = get_arg_val<uint32_t>(0);
    ptr[1] = get_arg_val<uint32_t>(1);
    ptr[2] = get_arg_val<uint32_t>(2);
    ptr[3] = get_arg_val<uint32_t>(3);
    ptr[4] = get_arg_val<uint32_t>(4);
}
