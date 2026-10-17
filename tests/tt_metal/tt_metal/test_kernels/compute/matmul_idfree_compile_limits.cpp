// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/2_0/matmul.h"

// Compiles the id-free (2.0) matmul operand-geometry contract; never executed on a device. The face grids of
// in0 / in1 and both face heights (tiny tiles, short faces) are compile-time args so the host test can probe
// which operand pairs the API accepts: only the pairs the Blackhole matmul LLK has programs for (in0 columns ==
// in1 rows, in0 not 16x16, short-face in0 a single face row, in1 one of 32x32 / 32x16 / 16x32 with 16-row
// faces) may compile. MM_THROTTLE (a define) and the math fidelity (ComputeConfig) are compile-time too: under
// throttle levels 4 / 5 at HiFi2 or higher only the 32x32 by 32x32 pair may compile.
void kernel_main() {
    constexpr std::uint32_t in0_faces_r = get_compile_time_arg_val(0);
    constexpr std::uint32_t in0_faces_c = get_compile_time_arg_val(1);
    constexpr std::uint32_t in1_faces_r = get_compile_time_arg_val(2);
    constexpr std::uint32_t in1_faces_c = get_compile_time_arg_val(3);
    constexpr bool block = get_compile_time_arg_val(4);
    constexpr std::uint32_t in0_face_r_dim = get_compile_time_arg_val(5);
    constexpr std::uint32_t in1_face_r_dim = get_compile_time_arg_val(6);

    using In0Op =
        experimental::LLKOperand<DataFormat::Float16_b, TensorShape{in0_face_r_dim, 16, in0_faces_r, in0_faces_c}>;
    using In1Op =
        experimental::LLKOperand<DataFormat::Float16_b, TensorShape{in1_face_r_dim, 16, in1_faces_r, in1_faces_c}>;

    // Never dereferenced: the program is only compiled.
    constexpr std::uint32_t l1_address = 0;
    constexpr std::uint32_t in0_tile_index = 0;
    constexpr std::uint32_t in1_tile_index = 0;
    constexpr std::uint32_t idst = 0;
    constexpr std::uint32_t ct_dim = 1;
    constexpr std::uint32_t rt_dim = 1;
    constexpr std::uint32_t kt_dim = 1;

    if constexpr (block) {
        experimental::matmul_block_init(In0Op(l1_address), In1Op(l1_address), 0 /*transpose*/, ct_dim, rt_dim, kt_dim);
        experimental::matmul_block(
            In0Op(l1_address), In1Op(l1_address), in0_tile_index, in1_tile_index, idst, ct_dim, rt_dim, kt_dim);
    } else {
        experimental::matmul_init(In0Op(l1_address), In1Op(l1_address));
        experimental::matmul_tiles(In0Op(l1_address), In1Op(l1_address), in0_tile_index, in1_tile_index, idst);
    }
}
