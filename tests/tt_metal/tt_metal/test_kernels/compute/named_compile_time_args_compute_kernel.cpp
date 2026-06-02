// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute-side test kernel for the experimental named ct_args:: feature.
//
// Covers the COMPUTE JIT compile path (TRISC_UNPACK / TRISC_MATH / TRISC_PACK)
// for named_args_generated.h. The data-movement path is covered by
// named_runtime_args_kernel.cpp. These reach the named-args header through
// different include chains (api/compute/common.h vs api/dataflow/dataflow_api.h),
// and the relocated genfiles emit + prolog #include must work in both. Before
// the relocation the compute path never received named_args_generated.h, so this
// kernel would fail to JIT-compile (ct_args:: undefined).
//
// Reads named compile-time args via the ct_args:: namespace and writes them to
// L1 at WRITE_ADDRESS (PACK only) so the host can verify the values.

#include <cstdint>

#include "api/compute/common.h"

void kernel_main() {
    // PACK is the only TRISC that populates the result slot; UNPACK/MATH no-op.
#ifdef TRISC_PACK
    volatile tt_l1_ptr uint32_t* l1_ptr = (volatile tt_l1_ptr uint32_t*)WRITE_ADDRESS;
    l1_ptr[0] = ct_args::my_kernel::param_a;
    l1_ptr[1] = ct_args::my_kernel::param_b;
#endif
}
