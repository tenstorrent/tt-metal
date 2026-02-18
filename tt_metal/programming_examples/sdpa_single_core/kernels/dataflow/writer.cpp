// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_iter = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // Generate a -inf tile for compute kernel's prev_max initialization.
    // Compute reads this via cb_wait_front (standard CB protocol, no semaphores needed).
    constexpr uint32_t cb_neginf = tt::CBIndex::c_7;
    cb_reserve_back(cb_neginf, 1);
    {
        uint32_t write_addr = get_write_ptr(cb_neginf);
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
        // Fill entire tile with 0xFF80FF80 (-inf in bf16, packed pairs)
        for (uint32_t i = 0; i < 2048 / sizeof(uint32_t); i++) {
            ptr[i] = 0xFF80FF80;
        }
    }
    cb_push_back(cb_neginf, 1);
}
