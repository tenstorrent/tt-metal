// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verifies that the force-included legacy CT map and the prolog-included
// named_args_generated.h work together on TRISC. Data-movement Blaze args are
// covered by blaze_named_runtime_args_kernel.cpp through a different include chain.
//
// Reads typed and legacy named compile-time args and writes them to L1 at
// WRITE_ADDRESS (PACK only) so the host can verify the values.

#include <cstdint>

#include "api/compute/common.h"

void kernel_main() {
    // PACK is the only TRISC that populates the result slot; UNPACK/MATH no-op.
#ifdef TRISC_PACK
    volatile tt_l1_ptr uint32_t* l1_ptr = (volatile tt_l1_ptr uint32_t*)WRITE_ADDRESS;
    l1_ptr[0] = blaze_ct_args::my_kernel::param_a;
    l1_ptr[1] = blaze_ct_args::my_kernel::param_b;
    l1_ptr[2] = get_named_compile_time_arg_val("legacy_param");
#endif
}
