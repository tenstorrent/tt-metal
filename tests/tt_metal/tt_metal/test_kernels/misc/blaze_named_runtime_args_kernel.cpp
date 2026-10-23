// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for named args (both CT and RT) with generated header.
// Reads named compile-time and runtime args, writes them to L1
// at WRITE_ADDRESS so the host can verify the values.

#include "api/dataflow/dataflow_api.h"
#include "experimental/blaze_named_args.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    // Named runtime args — dispatch type hidden by blaze_rt_args::get<>()
    l1_ptr[0] = blaze_rt_args::get<blaze_ct_args::my_kernel::marker>();
    l1_ptr[1] = blaze_rt_args::get<blaze_ct_args::my_kernel::core_idx>();

    // Named compile-time args — plain constexpr from blaze_ct_args:: namespace
    l1_ptr[2] = blaze_ct_args::my_kernel::param_a;
    l1_ptr[3] = blaze_ct_args::my_kernel::param_b;

    // The mixed-channel test requires the legacy lookup header alongside the Blaze header.
#ifdef TEST_LEGACY_NAMED_CT_ARGS
    l1_ptr[4] = get_named_compile_time_arg_val("legacy_param");
#endif
}
