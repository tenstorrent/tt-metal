// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "sanitizer/api.h"
#include "data_format_derive.h"
#include "experimental/2_0/llk_hw_configure.h"

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
// Fully typed inputs reconcile source exponent widths using the existing
// 2.0 format rules. Mixed CB/LLKOperand inputs retain the CB API's requirement
// that both configured source-register formats use the same exponent width.

namespace ckernel {

template <
    bool transpose = false,
    bool split_acc = false,
    bool dense_packing = false,
    bool fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void custom_mm_block_init(
    experimental::LLKOperand<F0, S0> /*in0*/,
    experimental::LLKOperand<F1, S1> /*in1*/,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    static_assert(experimental::is_legal_tile_shape(S0), "custom_mm_block: illegal in0 tile shape");
    static_assert(experimental::is_legal_tile_shape(S1), "custom_mm_block: illegal in1 tile shape");
    static_assert(
        S0.face_c_dim == 16 && S0.num_faces_r_dim == 1 && S0.num_faces_c_dim == 2 &&
            (S0.face_r_dim == 1 || S0.face_r_dim == 2 || S0.face_r_dim == 4 || S0.face_r_dim == 8),
        "custom_mm_block: in0 tile shape must be [1|2|4|8, 32]");
    static_assert(
        S1.face_r_dim == 16 && S1.face_c_dim == 16 && S1.num_faces_r_dim == 2 && S1.num_faces_c_dim == 2,
        "custom_mm_block: in1 tile shape must be [32, 32]");

    constexpr auto in0_descriptor = experimental::LLKOperand<F0, S0>::descriptor;
    constexpr auto in1_descriptor = experimental::LLKOperand<F1, S1>::descriptor;
    constexpr auto in1_register_format = infer_unpack_dst_format_2op<F1, F0>(fp32_dest_acc_en);

    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en, in1_descriptor, in0_descriptor>()));
    UNPACK((_llk_unpack_AB_custom_mm_init_<transpose>(
        S0.face_r_dim, static_cast<std::uint32_t>(in1_register_format), ct_dim)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    // Match the unpacker's physical ordering, including the source-format
    // caches maintained by current Metal: weights -> SrcA, activations -> SrcB.
    MATH((llk_math_hw_configure<fp32_dest_acc_en, in1_descriptor, in0_descriptor>()));
    MATH((_llk_math_custom_mm_init_<transpose, split_acc, dense_packing>(S0.face_r_dim, ct_dim)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, ckernel::PackMode::Default>(out_cb_id)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<ckernel::PackMode::Default, false>(out_cb_id)));
    PACK((_llk_pack_custom_mm_init_<dense_packing>()));
}

template <
    bool transpose = false,
    bool split_acc = false,
    bool dense_packing = false,
    bool fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat F1,
    TensorShape S1>
ALWI void custom_mm_block_init(
    const std::uint32_t in0_cb_id,
    experimental::LLKOperand<F1, S1> /*in1*/,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    static_assert(
        S1.face_r_dim == 16 && S1.face_c_dim == 16 && S1.num_faces_r_dim == 2 && S1.num_faces_c_dim == 2,
        "custom_mm_block: in1 tile shape must be [32, 32]");
    constexpr auto in1_register_format = infer_unpack_dst_format(F1, fp32_dest_acc_en);

    UNPACK(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_unpack_hw_configure_<fp32_dest_acc_en>(
            static_cast<std::uint32_t>(F1),
            unpack_src_format[in0_id],
            static_cast<std::uint32_t>(in1_register_format),
            unpack_dst_format[in0_id],
            S1.face_r_dim,
            get_operand_face_r_dim(in0_id),
            S1.total_num_faces(),
            get_operand_num_faces(in0_id),
            experimental::tile_stride_words(F1, S1),
            get_local_cb_interface(in0_id).fifo_page_size);
        _llk_unpack_AB_custom_mm_init_<transpose>(
            get_operand_face_r_dim(in0_id), static_cast<std::uint32_t>(in1_register_format), ct_dim);
    }));

    MATH(({
        const auto in0_id = get_operand_id(in0_cb_id);
        llk_math_pack_sync_init<fp32_dest_acc_en>();
        _llk_math_hw_configure_<fp32_dest_acc_en>(
            static_cast<std::uint32_t>(in1_register_format), unpack_dst_format[in0_id]);
        _llk_math_custom_mm_init_<transpose, split_acc, dense_packing>(get_operand_face_r_dim(in0_id), ct_dim);
    }));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, ckernel::PackMode::Default>(out_cb_id)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<ckernel::PackMode::Default, false>(out_cb_id)));
    PACK((_llk_pack_custom_mm_init_<dense_packing>()));
}

template <
    bool transpose = false,
    bool split_acc = false,
    bool dense_packing = false,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void custom_mm_block_init_short(
    experimental::LLKOperand<F0, S0> /*in0*/,
    experimental::LLKOperand<F1, S1> /*in1*/,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    static_assert(experimental::is_legal_tile_shape(S0), "custom_mm_block: illegal in0 tile shape");
    static_assert(experimental::is_legal_tile_shape(S1), "custom_mm_block: illegal in1 tile shape");
    static_assert(
        S0.face_c_dim == 16 && S0.num_faces_r_dim == 1 && S0.num_faces_c_dim == 2 &&
            (S0.face_r_dim == 1 || S0.face_r_dim == 2 || S0.face_r_dim == 4 || S0.face_r_dim == 8),
        "custom_mm_block: in0 tile shape must be [1|2|4|8, 32]");
    static_assert(
        S1.face_r_dim == 16 && S1.face_c_dim == 16 && S1.num_faces_r_dim == 2 && S1.num_faces_c_dim == 2,
        "custom_mm_block: in1 tile shape must be [32, 32]");

    constexpr auto in1_register_format = infer_unpack_dst_format_2op<F1, F0>(DST_ACCUM_MODE);
    UNPACK((_llk_unpack_AB_custom_mm_init_<transpose>(
        S0.face_r_dim, static_cast<std::uint32_t>(in1_register_format), ct_dim)));
    MATH((_llk_math_custom_mm_init_<transpose, split_acc, dense_packing>(S0.face_r_dim, ct_dim)));

    PACK((_llk_pack_custom_mm_init_<dense_packing>()));
}

template <bool transpose = false, bool split_acc = false, bool dense_packing = false, DataFormat F1, TensorShape S1>
ALWI void custom_mm_block_init_short(
    const std::uint32_t in0_cb_id, experimental::LLKOperand<F1, S1> /*in1*/, const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    static_assert(
        S1.face_r_dim == 16 && S1.face_c_dim == 16 && S1.num_faces_r_dim == 2 && S1.num_faces_c_dim == 2,
        "custom_mm_block: in1 tile shape must be [32, 32]");
    constexpr auto in1_register_format = infer_unpack_dst_format(F1, DST_ACCUM_MODE);
    UNPACK(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_unpack_AB_custom_mm_init_<transpose>(
            get_operand_face_r_dim(in0_id), static_cast<std::uint32_t>(in1_register_format), ct_dim);
    }));
    MATH(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_math_custom_mm_init_<transpose, split_acc, dense_packing>(get_operand_face_r_dim(in0_id), ct_dim);
    }));

    PACK((_llk_pack_custom_mm_init_<dense_packing>()));
}

template <
    bool finalize = true,
    bool read_transposed = false,
    bool clear_src = true,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void custom_mm_block(
    experimental::LLKOperand<F0, S0> in0,
    experimental::LLKOperand<F1, S1> in1,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    static_assert(experimental::is_legal_tile_shape(S0), "Illegal activation tile shape");
    static_assert(experimental::is_legal_tile_shape(S1), "Illegal weight tile shape");
    constexpr std::uint32_t in0_tile_size = experimental::tile_stride_words(F0, S0);
    constexpr std::uint32_t in1_tile_size = experimental::tile_stride_words(F1, S1);
    UNPACK((_llk_unpack_AB_custom_mm_<read_transposed, clear_src>(
        in1.l1_address, in0.l1_address, in1_tile_index, in0_tile_index, in1_tile_size, in0_tile_size, kt_dim, ct_dim)));
    MATH((_llk_math_custom_mm_<finalize>(S0.face_r_dim, dst_index, kt_dim, ct_dim)));
}

template <bool finalize = true, bool read_transposed = false, bool clear_src = true, DataFormat F1, TensorShape S1>
ALWI void custom_mm_block(
    const std::uint32_t in0_cb_id,
    experimental::LLKOperand<F1, S1> in1,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    constexpr std::uint32_t in1_tile_size = experimental::tile_stride_words(F1, S1);
    UNPACK(({
        const auto in0_id = get_operand_id(in0_cb_id);
        const auto& in0 = get_local_cb_interface(in0_id);
        _llk_unpack_AB_custom_mm_<read_transposed, clear_src>(
            in1.l1_address,
            in0.fifo_rd_ptr - 1,
            in1_tile_index,
            in0_tile_index,
            in1_tile_size,
            in0.fifo_page_size,
            kt_dim,
            ct_dim);
    }));
    MATH(({
        const auto in0_id = get_operand_id(in0_cb_id);
        _llk_math_custom_mm_<finalize>(get_operand_face_r_dim(in0_id), dst_index, kt_dim, ct_dim);
    }));
}

}  // namespace ckernel
