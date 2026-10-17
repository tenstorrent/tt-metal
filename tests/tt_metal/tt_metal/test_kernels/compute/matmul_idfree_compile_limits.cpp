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
// throttle levels 4 / 5 at HiFi2 or higher only the 32x32 by 32x32 pair may compile. `execute_only` inits with
// the 32x32 by 32x32 pair (supported at every throttle level) and hands the pair under test to the execute alone,
// so the host test can pin that matmul_tiles / matmul_block reject an unsupported pair on their own: the execute
// deduces its own operands and configures the unpacker from them.

namespace {

// The documented Blackhole matmul support matrix, as (in0 face grid, in0 face height, in1 face grid; in1 faces are
// always 16 rows): the 32x32 in0 or any single-face-row 32-column in0 (face height 1..16) against a 32x32 or 32x16
// in1, and the 32x16 in0 against the 16x32 in1. 13 pairs.
constexpr bool in_support_matrix(TensorShape in0, TensorShape in1) {
    const bool in1_32x32_or_32x16 = in1.num_faces_r_dim == 2 && (in1.num_faces_c_dim == 2 || in1.num_faces_c_dim == 1);
    const bool in0_32x32 =
        in0.face_r_dim == ckernel::MAX_FACE_R_DIM && in0.num_faces_r_dim == 2 && in0.num_faces_c_dim == 2;
    const bool in0_one_face_row_32_wide = in0.num_faces_r_dim == 1 && in0.num_faces_c_dim == 2;
    const bool in0_32x16 =
        in0.face_r_dim == ckernel::MAX_FACE_R_DIM && in0.num_faces_r_dim == 2 && in0.num_faces_c_dim == 1;
    const bool in1_16x32 = in1.num_faces_r_dim == 1 && in1.num_faces_c_dim == 2;
    return in1.face_r_dim == ckernel::MAX_FACE_R_DIM && in0.face_c_dim == ckernel::MAX_FACE_C_DIM &&
           in1.face_c_dim == ckernel::MAX_FACE_C_DIM &&
           ((in1_32x32_or_32x16 && (in0_32x32 || in0_one_face_row_32_wide)) || (in0_32x16 && in1_16x32));
}

// Over every legal tile shape pair (face height 1 / 2 / 4 / 8 / 16, face grid 1..2 x 1..2 per operand: 400 pairs)
// the API's predicate -- the LLK's validate_matmul_tensor_shapes_ minus the 16x16 by 16x16 pair -- must accept
// exactly the support matrix above, which is also what the matmul_init / matmul_tiles assert messages promise.
constexpr bool predicate_matches_support_matrix() {
    constexpr std::uint8_t face_heights[] = {1, 2, 4, 8, ckernel::MAX_FACE_R_DIM};
    constexpr std::uint8_t face_counts[] = {1, 2};
    std::uint32_t pairs = 0;
    std::uint32_t supported = 0;
    for (const std::uint8_t fr0 : face_heights) {
        for (const std::uint8_t nr0 : face_counts) {
            for (const std::uint8_t nc0 : face_counts) {
                for (const std::uint8_t fr1 : face_heights) {
                    for (const std::uint8_t nr1 : face_counts) {
                        for (const std::uint8_t nc1 : face_counts) {
                            const TensorShape in0{fr0, ckernel::MAX_FACE_C_DIM, nr0, nc0};
                            const TensorShape in1{fr1, ckernel::MAX_FACE_C_DIM, nr1, nc1};
                            if (!experimental::is_legal_tile_shape(in0) || !experimental::is_legal_tile_shape(in1)) {
                                return false;
                            }
                            ++pairs;
                            const bool expected = in_support_matrix(in0, in1);
                            supported += expected;
                            if (experimental::is_supported_matmul_operand_pair(in0, in1) != expected) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    return pairs == 400 && supported == 13;
}
static_assert(
    predicate_matches_support_matrix(),
    "is_supported_matmul_operand_pair must accept exactly the documented Blackhole matmul support matrix");

}  // namespace

void kernel_main() {
    constexpr std::uint32_t in0_faces_r = get_compile_time_arg_val(0);
    constexpr std::uint32_t in0_faces_c = get_compile_time_arg_val(1);
    constexpr std::uint32_t in1_faces_r = get_compile_time_arg_val(2);
    constexpr std::uint32_t in1_faces_c = get_compile_time_arg_val(3);
    constexpr bool block = get_compile_time_arg_val(4);
    constexpr std::uint32_t in0_face_r_dim = get_compile_time_arg_val(5);
    constexpr std::uint32_t in1_face_r_dim = get_compile_time_arg_val(6);
    constexpr bool execute_only = get_compile_time_arg_val(7);

    using In0Op = experimental::LLKOperand<
        DataFormat::Float16_b,
        TensorShape{in0_face_r_dim, ckernel::MAX_FACE_C_DIM, in0_faces_r, in0_faces_c}>;
    using In1Op = experimental::LLKOperand<
        DataFormat::Float16_b,
        TensorShape{in1_face_r_dim, ckernel::MAX_FACE_C_DIM, in1_faces_r, in1_faces_c}>;
    // The pair the execute-only mode inits with: 32x32 by 32x32, supported at every throttle level.
    using FullTileOp = experimental::LLKOperand<DataFormat::Float16_b, DEFAULT_TENSOR_SHAPE>;

    // Never dereferenced: the program is only compiled.
    constexpr std::uint32_t l1_address = 0;
    constexpr std::uint32_t in0_tile_index = 0;
    constexpr std::uint32_t in1_tile_index = 0;
    constexpr std::uint32_t idst = 0;
    constexpr std::uint32_t ct_dim = 1;
    constexpr std::uint32_t rt_dim = 1;
    constexpr std::uint32_t kt_dim = 1;

    if constexpr (block) {
        if constexpr (execute_only) {
            experimental::matmul_block_init(
                FullTileOp(l1_address), FullTileOp(l1_address), 0 /*transpose*/, ct_dim, rt_dim, kt_dim);
        } else {
            experimental::matmul_block_init(
                In0Op(l1_address), In1Op(l1_address), 0 /*transpose*/, ct_dim, rt_dim, kt_dim);
        }
        experimental::matmul_block(
            In0Op(l1_address), In1Op(l1_address), in0_tile_index, in1_tile_index, idst, ct_dim, rt_dim, kt_dim);
    } else {
        if constexpr (execute_only) {
            experimental::matmul_init(FullTileOp(l1_address), FullTileOp(l1_address));
        } else {
            experimental::matmul_init(In0Op(l1_address), In1Op(l1_address));
        }
        experimental::matmul_tiles(In0Op(l1_address), In1Op(l1_address), in0_tile_index, in1_tile_index, idst);
    }
}
