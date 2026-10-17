// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/2_0/matmul.h"

// Compiles the id-free (2.0) matmul operand-geometry contract; never executed on a device. The face grids of
// in0 / in1 are compile-time args so the host test can probe which operand pairs the API accepts: a single-face
// in0 by a single-face in1 (16x16 by 16x16) has no dedicated math program and must be rejected at compile time.
void kernel_main() {
    constexpr std::uint32_t in0_faces_r = get_compile_time_arg_val(0);
    constexpr std::uint32_t in0_faces_c = get_compile_time_arg_val(1);
    constexpr std::uint32_t in1_faces_r = get_compile_time_arg_val(2);
    constexpr std::uint32_t in1_faces_c = get_compile_time_arg_val(3);
    constexpr bool block = get_compile_time_arg_val(4);

    using In0Op = experimental::LLKOperand<DataFormat::Float16_b, TensorShape{16, 16, in0_faces_r, in0_faces_c}>;
    using In1Op = experimental::LLKOperand<DataFormat::Float16_b, TensorShape{16, 16, in1_faces_r, in1_faces_c}>;

    if constexpr (block) {
        experimental::matmul_block_init(In0Op(0), In1Op(0), 0 /*transpose*/, 1 /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/);
        experimental::matmul_block(In0Op(0), In1Op(0), 0, 0, 0 /*idst*/, 1 /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/);
    } else {
        experimental::matmul_init(In0Op(0), In1Op(0));
        experimental::matmul_tiles(In0Op(0), In1Op(0), 0, 0, 0 /*idst*/);
    }
}
