// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR: persistent owner compute through the engine-callable phase (mk_ar_reduce.hpp), MK_PS_RED=hdr.
// Same compile args as mk_ps_reduction.cpp (SRED not supported here).
#include <cstdint>
#define MKAR_CB_RED get_compile_time_arg_val(0)
#define MKAR_CB_OUT get_compile_time_arg_val(1)
#define MKAR_FUSE_RES (get_compile_time_arg_val(2) != 0)
#define MKAR_CB_RES get_compile_time_arg_val(3)
#define MKAR_ITERS get_compile_time_arg_val(4)
#define MKAR_STATS (get_compile_time_arg_val(5) != 0)
#define MKAR_CB_SQ get_compile_time_arg_val(7)
#define MKAR_CB_PART get_compile_time_arg_val(8)
#define MKAR_CB_SCALER get_compile_time_arg_val(9)
#include "mk_ar_reduce.hpp"

static mkar::Reduce red;

void kernel_main() {
    if (!red.init(0)) {
        return;
    }
    compute_kernel_hw_startup(MKAR_CB_RED, MKAR_CB_RED, MKAR_CB_OUT);
    for (uint32_t it = 0; it < MKAR_ITERS; ++it) {
        red.iter();
    }
}
