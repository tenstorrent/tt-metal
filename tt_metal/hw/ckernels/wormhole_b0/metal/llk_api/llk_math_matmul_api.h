// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    // In0/operandA -> srcB, In1/operandB -> srcA
    SAN_HOOK(init<OperationFpuMatmul>(
        StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
        StateVal<OperationFpuMatmul::ThrottleLevel>(THROTTLE_LEVEL),
        StateVal<OperationFpuMatmul::CtDim>(ct_dim),
        StateVal<OperationFpuMatmul::RtDim>(rt_dim),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[in1_id]),
        StateDiscard<std::uint32_t>(transpose),
        StateDiscard<std::uint32_t>(in0_tile_r_dim),
        StateDiscard<std::uint32_t>(in0_tile_c_dim),
        StateDiscard<std::uint32_t>(in1_tile_r_dim),
        StateDiscard<std::uint32_t>(in1_tile_c_dim),
        StateDiscard<bool>(partial_face)));

    _llk_math_matmul_init_<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0, uint32_t num_faces = 4 /*not used*/>
inline void llk_math_matmul(const uint dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    static_assert(num_faces == 4, "num_faces other than 4 is not supported in llk_math_matmul");
    LLK_ASSERT(
        (ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim) <
         get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "llk_math_matmul: computed matmul dest tile range exceeds available dest register "
        "capacity. Uncomment the DPRINT block below and enable DPRINT support to inspect "
        "the calculated and max dest tile values.");

    // DPRINT("llk_math_matmul: calculated dest tiles = {}, max dest tiles = {} (dst_index={}, ct_dim={},
    // rt_dim={})\n",
    //     ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim),
    //     get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>(),
    //     dst_index,
    //     ct_dim,
    //     rt_dim);

    SAN_HOOK(execute<OperationFpuMatmul>(
        StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
        StateVal<OperationFpuMatmul::ThrottleLevel>(THROTTLE_LEVEL),
        StateVal<OperationFpuMatmul::CtDim>(ct_dim),
        StateVal<OperationFpuMatmul::RtDim>(rt_dim),
        StateDiscard<uint>(dst_index)));

    _llk_math_matmul_<math_fidelity, THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim);
}

// K-looping variants: kt_dim K-tile steps in one call; for a 2x2 block the LLK alternates the MVMUL order between
// steps (see _llk_math_matmul_kloop_). Same operands, dst layout and numerics as llk_math_matmul.
template <MathFidelity math_fidelity>
inline void llk_math_matmul_kloop_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    SAN_HOOK(init<OperationFpuMatmul>(
        StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
        StateVal<OperationFpuMatmul::ThrottleLevel>(0),
        StateVal<OperationFpuMatmul::CtDim>(ct_dim),
        StateVal<OperationFpuMatmul::RtDim>(rt_dim),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[in1_id]),
        StateDiscard<std::uint32_t>(transpose),
        StateDiscard<std::uint32_t>(in0_tile_r_dim),
        StateDiscard<std::uint32_t>(in0_tile_c_dim),
        StateDiscard<std::uint32_t>(in1_tile_r_dim),
        StateDiscard<std::uint32_t>(in1_tile_c_dim),
        StateDiscard<bool>(partial_face)));

    _llk_math_matmul_kloop_init_<math_fidelity>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity>
inline void llk_math_matmul_kloop(
    const uint dst_index, const std::uint32_t ct_dim, const std::uint32_t rt_dim, const std::uint32_t kt_dim) {
    LLK_ASSERT(
        (ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim) <
         get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "llk_math_matmul_kloop: computed matmul dest tile range exceeds available dest register capacity");

    for (std::uint32_t k = 0; k < kt_dim; k++) {
        SAN_HOOK(execute<OperationFpuMatmul>(
            StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
            StateVal<OperationFpuMatmul::ThrottleLevel>(0),
            StateVal<OperationFpuMatmul::CtDim>(ct_dim),
            StateVal<OperationFpuMatmul::RtDim>(rt_dim),
            StateDiscard<uint>(dst_index)));
    }

    _llk_math_matmul_kloop_<math_fidelity>(dst_index, ct_dim, rt_dim, kt_dim);
}
