// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/2_0/matmul.h"

// Compiles the id-free (2.0) matmul operand-geometry contract; never executed on a device. The face grids of
// in0 / in1 (and the in0 face height, for tiny tiles) are compile-time args so the host test can probe which
// operand pairs the API accepts: only the pairs the Blackhole matmul LLK has programs for (in0 columns == in1
// rows, in0 not 16x16, in1 one of 32x32 / 32x16 / 16x32) may compile.
void kernel_main() {
    constexpr std::uint32_t in0_faces_r = get_compile_time_arg_val(0);
    constexpr std::uint32_t in0_faces_c = get_compile_time_arg_val(1);
    constexpr std::uint32_t in1_faces_r = get_compile_time_arg_val(2);
    constexpr std::uint32_t in1_faces_c = get_compile_time_arg_val(3);
    constexpr bool block = get_compile_time_arg_val(4);
    constexpr std::uint32_t in0_face_r_dim = get_compile_time_arg_val(5);

    using In0Op =
        experimental::LLKOperand<DataFormat::Float16_b, TensorShape{in0_face_r_dim, 16, in0_faces_r, in0_faces_c}>;
    using In1Op = experimental::LLKOperand<DataFormat::Float16_b, TensorShape{16, 16, in1_faces_r, in1_faces_c}>;

    constexpr std::uint32_t in0_tile_index = 0;
    constexpr std::uint32_t in1_tile_index = 0;
    constexpr std::uint32_t idst = 0;

    if constexpr (block) {
        experimental::matmul_block_init(In0Op(0), In1Op(0), 0 /*transpose*/, 1 /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/);
        experimental::matmul_block(
            In0Op(0), In1Op(0), in0_tile_index, in1_tile_index, idst, 1 /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/);
    } else {
        experimental::matmul_init(In0Op(0), In1Op(0));
        experimental::matmul_tiles(In0Op(0), In1Op(0), in0_tile_index, in1_tile_index, idst);
    }
}
