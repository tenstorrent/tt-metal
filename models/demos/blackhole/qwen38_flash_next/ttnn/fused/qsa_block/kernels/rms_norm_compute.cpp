// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The rms_norm mirror alone (the LLK pin tool): CBs as rms_norm_mirror.h with in 0, scaler 1, eps 2, gamma 3,
// x^2 4, var 5, rsqrt 6, unit 7, out 16.  Compile-time args: 0 Wt, 1 W.  Runtime arg 0: rows.

#include "rms_norm_mirror.h"
#include "../../kernels/zones.h"

void kernel_main() {
    FUSED_ZONE("fz_qs_rms_c_main");
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    const uint32_t rows = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 0, 4);
    rms_norm_rows<Wt, W, 0, 1, 2, 3, 4, 5, 6, 7, 16>(rows);
    DataflowBuffer scaler(1);
    scaler.pop_front(1);
}
