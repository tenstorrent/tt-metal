// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/experimental/compressed_custom_mm.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "sanitizer/api.h"

// LLKOperand overloads live in this opt-in header, like 2_0/hw_startup.h:
// pulling ckernel::experimental into legacy kernels can collide with the
// top-level experimental namespace used by kernel_args.h.
//
// These retain the CB API's synchronization, layout and LoFi contracts. L1
// operand addresses are in 16-byte words, using the same base convention as
// cb_read_address(). The caller owns input readiness and output CB capacity.
// Short init configures only the operation: the caller must first configure
// hardware formats/geometry and destination synchronization. It must quiesce
// prior operations before reconfiguration. Output packing remains CB-based.
// Metadata uses the existing compressed-custom-mm byte-addressed format and
// selects BFP8_b/BFP4_b/BFP2_b or skipped tiles. The operand identifies the
// compressed stream base; metadata determines each weight tile's stride.
// Like the CB path, split accumulation and finalization remain disabled.

namespace ckernel {

template <
    bool transpose = false,
    bool split_acc = false,
    bool dense_packing = false,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void compressed_custom_mm_block_init_short(
    experimental::LLKOperand<F0, S0> /*in0*/, experimental::LLKOperand<F1, S1> /*in1*/) {
    SAN_HOOK(unsupported());
    static_assert(experimental::is_legal_tile_shape(S0), "Illegal activation tile shape");
    static_assert(experimental::is_legal_tile_shape(S1), "Illegal weight tile shape");
    static_assert(
        S0.face_c_dim == 16 && S0.num_faces_r_dim == 1 && S0.num_faces_c_dim == 2 &&
            (S0.face_r_dim == 1 || S0.face_r_dim == 2 || S0.face_r_dim == 4 || S0.face_r_dim == 8),
        "compressed_custom_mm: in0 tile shape must be [1|2|4|8, 32]");
    static_assert(
        S1.face_r_dim == 16 && S1.face_c_dim == 16 && S1.num_faces_r_dim == 2 && S1.num_faces_c_dim == 2,
        "compressed_custom_mm: in1 tile shape must be [32, 32]");
    UNPACK((_llk_unpack_AB_compressed_custom_mm_init_<transpose>(S0.face_r_dim)));
    MATH((_llk_math_compressed_custom_mm_init_<transpose, false, dense_packing>(S0.face_r_dim)));
    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

template <bool transpose = false, bool split_acc = false, bool dense_packing = false, DataFormat F1, TensorShape S1>
ALWI void compressed_custom_mm_block_init_short(
    const std::uint32_t in0_cb_id, experimental::LLKOperand<F1, S1> /*in1*/) {
    SAN_HOOK(unsupported());
    static_assert(
        S1.face_r_dim == 16 && S1.face_c_dim == 16 && S1.num_faces_r_dim == 2 && S1.num_faces_c_dim == 2,
        "compressed_custom_mm: in1 tile shape must be [32, 32]");
    UNPACK(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_unpack_AB_compressed_custom_mm_init_<transpose>(get_operand_face_r_dim(in0_id));
    }));
    MATH(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_math_compressed_custom_mm_init_<transpose, false, dense_packing>(get_operand_face_r_dim(in0_id));
    }));
    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

template <bool finalize = true, bool clear_src = true, DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void compressed_custom_mm_block(
    experimental::LLKOperand<F0, S0> in0,
    experimental::LLKOperand<F1, S1> in1,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    static_assert(experimental::is_legal_tile_shape(S0), "Illegal activation tile shape");
    static_assert(experimental::is_legal_tile_shape(S1), "Illegal weight tile shape");
    UNPACK((_llk_unpack_AB_compressed_custom_mm_<clear_src>(
        in1.l1_address, in0.l1_address, base_address_meta, kt_dim, ct_dim)));
    MATH((_llk_math_compressed_custom_mm_<false>(base_address_meta, S0.face_r_dim, dst_index, kt_dim, ct_dim)));
}

template <bool finalize = true, bool clear_src = true, DataFormat F1, TensorShape S1>
ALWI void compressed_custom_mm_block(
    const std::uint32_t in0_cb_id,
    experimental::LLKOperand<F1, S1> in1,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    UNPACK(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_unpack_AB_compressed_custom_mm_<clear_src>(
            in1.l1_address, get_local_cb_interface(in0_id).fifo_rd_ptr - 1, base_address_meta, kt_dim, ct_dim);
    }));
    MATH(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_math_compressed_custom_mm_<false>(
            base_address_meta, get_operand_face_r_dim(in0_id), dst_index, kt_dim, ct_dim);
    }));
}

}  // namespace ckernel
