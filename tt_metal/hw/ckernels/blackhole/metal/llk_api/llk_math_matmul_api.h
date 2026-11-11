// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

// Unified core, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). Matmul math is
// FORMAT-FREE: it consumes only tile geometry (tile r/c dims + partial_face). The per-source prologue
// (resolving these from a CB id, or from an LLKMemDescriptor) lives in the callers. The matmul math EXECUTE
// (llk_math_matmul below) already takes no operand, so it is reused directly by the id-free API. row_mop selects one
// MOP per reuse row for full 32x32 tiles (see _llk_math_matmul_init_); tile_mop keeps one MOP per output tile instead
// of the default row form (see llk_math_matmul_tile_mop). Init and execute must pass the same values. in1_order is the
// DEST row group order of in1 32x16 tiles (see _llk_math_matmul_init_), chosen by the kernel at compile time.
template <
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0,
    bool row_mop = false,
    bool tile_mop = true,
    int in1_order = 0,
    std::uint32_t reuse_dir = 0,
    bool row_end_fold = false,
    bool half_end_fold = false,
    std::uint32_t key_ct = 0,
    std::uint32_t key_rt = 0>
inline void llk_math_matmul_init_impl(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim) {
    // an unkeyed caller (no block keys) gets main's init exactly: tile MOP, no end fold, the bias word written; the keyed forms keep their opt-ins
    _llk_math_matmul_init_<math_fidelity, THROTTLE_LEVEL, row_mop, tile_mop, in1_order, !(DST_SYNC_MODE == DstSync::SyncHalf && DST_ACCUM_MODE), (DST_SYNC_MODE == DstSync::SyncFull && key_ct != 0 && key_rt != 0) || half_end_fold, reuse_dir, row_end_fold, key_ct, key_rt, (key_ct == 0 || key_rt == 0)>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

// The tile_mop argument for a kernel's k steps per DEST section and output format; 0 and Invalid when unknown.
template <bool row_mop, std::uint32_t kt_per_section, DataFormat out_format>
constexpr bool llk_math_matmul_tile_mop() {
    return !row_mop &&
           !matmul_math_row_form<MATH_FIDELITY, DST_SYNC_MODE, DST_ACCUM_MODE, kt_per_section, out_format>();
}

// The half_end_fold argument for a kernel's math fidelity, k steps per DEST section, operand and output formats (see matmul_half_end_fold);
// kernels that pass no formats keep the end op. block_rt, block_ct and block_kt (optional keys of the caller, 0 keep today's code) name the block,
// for the cells where the fold reads slower (matmul_end_fold_off_cells).
template <
    bool row_mop,
    MathFidelity math_fidelity,
    std::uint32_t kt_per_section,
    DataFormat out_format,
    DataFormat in0 = DataFormat::Invalid,
    DataFormat in1 = DataFormat::Invalid,
    std::uint32_t block_rt = 0,
    std::uint32_t block_ct = 0,
    std::uint32_t block_kt = 0>
constexpr bool llk_math_matmul_half_end_fold() {
    if constexpr (block_rt != 0 && block_ct != 0 && block_kt != 0 && in0 != DataFormat::Invalid && in1 != DataFormat::Invalid) {
        return llk_math_matmul_tile_mop<row_mop, kt_per_section, out_format>() &&
               matmul_half_end_fold_block<math_fidelity, DST_SYNC_MODE, DST_ACCUM_MODE, kt_per_section, out_format, in0, in1, block_rt, block_ct, block_kt>();
    } else {
        return llk_math_matmul_tile_mop<row_mop, kt_per_section, out_format>() &&
               matmul_half_end_fold<math_fidelity, DST_SYNC_MODE, DST_ACCUM_MODE, kt_per_section, out_format, in0, in1>();
    }
}


template <
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0,
    bool row_mop = false,
    bool tile_mop = true,
    int in1_order = 0,
    std::uint32_t reuse_dir = 0,
    bool row_end_fold = false,
    bool half_end_fold = false,
    std::uint32_t key_ct = 0,
    std::uint32_t key_rt = 0>
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

    llk_math_matmul_init_impl<math_fidelity, THROTTLE_LEVEL, row_mop, tile_mop, in1_order, reuse_dir, row_end_fold, half_end_fold, key_ct, key_rt>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

// the init of a throttle switch (_llk_math_matmul_init_switch_), for matmul_block_math_dynamic_throttle only
template <
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0,
    bool row_mop = false,
    bool tile_mop = true,
    int in1_order = 0,
    std::uint32_t reuse_dir = 0,
    bool row_end_fold = false,
    bool half_end_fold = false,
    std::uint32_t key_ct = 0,
    std::uint32_t key_rt = 0,
    bool light_switch = false>
inline void llk_math_matmul_init_switch_impl(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim) {
    if constexpr (light_switch) {
        // only for a caller that guarantees no op between the matmul inits rewrote ADDR_MOD 0, 1, 2 and 4: writes ADDR_MOD_5 and 6, the replay and the MOP
        _llk_math_matmul_init_switch_<math_fidelity, THROTTLE_LEVEL, row_mop, tile_mop, in1_order, !(DST_SYNC_MODE == DstSync::SyncHalf && DST_ACCUM_MODE), (DST_SYNC_MODE == DstSync::SyncFull && key_ct != 0 && key_rt != 0) || half_end_fold, reuse_dir, row_end_fold, key_ct, key_rt, (key_ct == 0 || key_rt == 0)>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
    } else {
        // main's switch: the full init again
        llk_math_matmul_init_impl<math_fidelity, THROTTLE_LEVEL, row_mop, tile_mop, in1_order, reuse_dir, row_end_fold, half_end_fold, key_ct, key_rt>(
            in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
    }
}

template <
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0,
    bool row_mop = false,
    bool tile_mop = true,
    int in1_order = 0,
    std::uint32_t reuse_dir = 0,
    bool row_end_fold = false,
    bool half_end_fold = false,
    std::uint32_t key_ct = 0,
    std::uint32_t key_rt = 0,
    bool light_switch = false>
inline void llk_math_matmul_init_switch(
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

    llk_math_matmul_init_switch_impl<math_fidelity, THROTTLE_LEVEL, row_mop, tile_mop, in1_order, reuse_dir, row_end_fold, half_end_fold, key_ct, key_rt, light_switch>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0,
    std::uint32_t num_faces = 4 /*not used*/,
    bool row_mop = false,
    bool tile_mop = true,
    std::uint32_t reuse_dir = 0,
    bool row_end_fold = false>
inline void llk_math_matmul(
    const std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    static_assert(num_faces == 4, "num_faces other than 4 is not supported in llk_math_matmul");

    // DPRINT("llk_math_matmul: calculated dest tiles = {}, max dest tiles = {} (dst_index={}, ct_dim={},
    // rt_dim={})\n",
    //     ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim),
    //     get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>(),
    //     dst_index,
    //     ct_dim,
    //     rt_dim);

    LLK_ASSERT(
        (ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim) <
         get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "llk_math_matmul: computed matmul dest tile range exceeds available dest register "
        "capacity. Uncomment the DPRINT block above and enable DPRINT support to inspect "
        "the calculated and max dest tile values.");

    SAN_HOOK(execute<OperationFpuMatmul>(
        StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
        StateVal<OperationFpuMatmul::ThrottleLevel>(THROTTLE_LEVEL),
        StateVal<OperationFpuMatmul::CtDim>(ct_dim),
        StateVal<OperationFpuMatmul::RtDim>(rt_dim),
        StateDiscard<uint>(dst_index)));

    _llk_math_matmul_<math_fidelity, THROTTLE_LEVEL, row_mop, tile_mop, reuse_dir, row_end_fold>(dst_index, ct_dim, rt_dim);
}
