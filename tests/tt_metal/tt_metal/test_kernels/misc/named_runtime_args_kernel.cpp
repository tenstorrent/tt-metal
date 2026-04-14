// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for named args (both CT and RT) with generated header.
// Reads named compile-time and runtime args, writes them to L1
// at WRITE_ADDRESS so the host can verify the values.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    // Named runtime args — dispatch type hidden by rt_args::get<>()
    l1_ptr[0] = rt_args::get<rt_args::my_kernel::marker>();
    l1_ptr[1] = rt_args::get<rt_args::my_kernel::core_idx>();

    // Named compile-time args — plain constexpr from ct_args:: namespace
    l1_ptr[2] = ct_args::my_kernel::param_a;
    l1_ptr[3] = ct_args::my_kernel::param_b;
}
