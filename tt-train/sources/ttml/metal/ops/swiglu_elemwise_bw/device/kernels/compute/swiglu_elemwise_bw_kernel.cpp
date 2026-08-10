// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SwiGLU elemwise backward on two separate [.,I] tensors (linear1 = silu'd branch, gate = plain).
// Per-block math shared with swiglu_packed_bw via swiglu_gate_bw_compute.hpp.

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "tt-train/sources/ttml/metal/common/swiglu_gate_bw_compute.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

constexpr uint32_t cb_linear1 = tt::CBIndex::c_0;  // gate branch (silu'd)
constexpr uint32_t cb_gate = tt::CBIndex::c_1;     // plain branch
constexpr uint32_t cb_dL_dprod = tt::CBIndex::c_2;
constexpr uint32_t cb_dL_dlinear1 = tt::CBIndex::c_3;
constexpr uint32_t cb_dL_dgate = tt::CBIndex::c_4;
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_5;
constexpr uint32_t cb_scratch = tt::CBIndex::c_6;
constexpr uint32_t cb_silu_grad = tt::CBIndex::c_7;

void kernel_main() {
    init_sfpu(cb_linear1, cb_dL_dlinear1);
    // TODO(#52395): compute_kernel_hw_startup is a call-once API and should be the kernel's first Tensix-engine call, but here it follows another engine op (init_sfpu / a prior startup); see the issue.
    compute_kernel_hw_startup(cb_linear1, cb_gate, cb_dL_dlinear1);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_linear1, block_size);
            cb_wait_front(cb_gate, block_size);
            cb_wait_front(cb_dL_dprod, block_size);

            // linear1 is the silu'd (gate) branch; cb_gate here is the plain branch.
            swiglu_gate_bw_block<
                cb_linear1,
                cb_gate,
                cb_dL_dprod,
                cb_dL_dlinear1,
                cb_dL_dgate,
                cb_sigmoid,
                cb_scratch,
                cb_silu_grad,
                block_size>();

            cb_pop_front(cb_linear1, block_size);
            cb_pop_front(cb_gate, block_size);
            cb_pop_front(cb_dL_dprod, block_size);
        }
    }
}
