// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed-SwiGLU backward: reads gate|up from the packed forward input, writes dgate|dup into
// the two halves of one dpacked tensor. Per-block math shared with swiglu_elemwise_bw (see header).

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "tt-train/sources/ttml/metal/common/swiglu_gate_bw_compute.hpp"

constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

constexpr uint32_t cb_gate = tt::CBIndex::c_0;   // gate branch (silu'd)
constexpr uint32_t cb_up = tt::CBIndex::c_1;     // up branch
constexpr uint32_t cb_dh = tt::CBIndex::c_2;     // upstream grad dL/dh
constexpr uint32_t cb_dgate = tt::CBIndex::c_3;  // out: grad wrt gate branch
constexpr uint32_t cb_dup = tt::CBIndex::c_4;    // out: grad wrt up branch
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_5;
constexpr uint32_t cb_scratch = tt::CBIndex::c_6;
constexpr uint32_t cb_silu_grad = tt::CBIndex::c_7;

void kernel_main() {
    compute_kernel_hw_startup(cb_gate, cb_up, cb_dgate);
    copy_init(cb_gate);

    for (uint32_t block = 0; block < num_blocks_per_core; ++block) {
        cb_wait_front(cb_gate, block_size);
        cb_wait_front(cb_up, block_size);
        cb_wait_front(cb_dh, block_size);

        swiglu_gate_bw_block<
            cb_gate,
            cb_up,
            cb_dh,
            cb_dgate,
            cb_dup,
            cb_sigmoid,
            cb_scratch,
            cb_silu_grad,
            block_size>();

        cb_pop_front(cb_gate, block_size);
        cb_pop_front(cb_up, block_size);
        cb_pop_front(cb_dh, block_size);
    }
}
