// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "experimental/llk_math_mul_reduce_scalar.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Fused multiply and scalar reduction
 *************************************************************************/

template <MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar_init(
    const std::uint32_t operand_A, const std::uint32_t acc_to_dest = 0) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_eltwise_binary_init_<
        EltwiseBinaryType::ELWMUL,
        BroadcastType::NONE,
        math_fidelity,
        EltwiseBinaryReuseDestType::NONE>(tensor_shape, acc_to_dest);
}

/**
 * @tparam share_slots: Two 16x32 products share a DEST slot, so dst_index is a 32-row product index: product i goes to
 *         rows 32 * (i % 2) of slot i / 2. Other shapes keep one tile per slot.
 * @note Move products written with share_slots through @ref llk_math_mul_reduce_scalar_move_product with share_slots
 * too.
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool share_slots = false>
inline void llk_math_eltwise_mul_reduce_scalar(
    std::uint32_t dst_index, const std::uint32_t icb0, const bool clear_fp32_dst_acc = true) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(icb0);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    if constexpr (share_slots) {
        if (_llk_math_mul_reduce_scalar_shares_slot_(tensor_shape)) {
            _llk_math_mul_reduce_scalar_mul_half_slot_<math_fidelity>(tensor_shape, dst_index);
            return;
        }
    }

    _llk_math_eltwise_binary_<
        EltwiseBinaryType::ELWMUL,
        BroadcastType::NONE,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        EltwiseBinaryReuseDestType::NONE>(tensor_shape, dst_index, clear_fp32_dst_acc);
}

template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void llk_math_mul_reduce_scalar_reduce_init() {
    SAN_HOOK(unsupported());
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, math_fidelity, enforce_fp32_accumulation>();
}

template <MathFidelity math_fidelity, bool tile_setup = true>
inline void llk_math_mul_reduce_column(const std::uint32_t dst_index, const std::uint32_t icb0) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(icb0);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    _llk_math_mul_reduce_column_<math_fidelity, tile_setup>(dst_index, tensor_shape);
}

/**
 * @brief Zero face 0 of dest[dst_index], where the column passes pool; see @ref
 * _llk_math_mul_reduce_scalar_clear_pool_face_.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_math_mul_reduce_scalar_clear_pool_face(const std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    _llk_math_mul_reduce_scalar_clear_pool_face_<is_fp32_dest_acc_en>(dst_index);
}

template <MathFidelity math_fidelity>
inline void llk_math_mul_reduce_scalar() {
    SAN_HOOK(unsupported());
    _llk_math_mul_reduce_scalar_<math_fidelity>();
}

/**
 * @brief Restore the multiply's address modifiers after a reduce phase; the reduce does not touch the multiply's MOP.
 */
template <MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar_reinit() {
    SAN_HOOK(unsupported());
    eltwise_binary_configure_addrmod<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, math_fidelity>();
}

inline void llk_math_mul_reduce_scalar_clear_dvalid() {
    SAN_HOOK(unsupported());
    _llk_math_mul_reduce_scalar_clear_dvalid_();
}

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_mul_reduce_scalar_move_dest_to_src(std::uint32_t idst = 0) {
    SAN_HOOK(unsupported());
    _llk_math_mul_reduce_scalar_move_dest_to_src_<binary_reuse_dest>(idst);
}

/**
 * @brief The move into SrcA before product i's column pass.
 *
 * @tparam share_slots: As given to @ref llk_math_eltwise_mul_reduce_scalar for the products.
 */
template <bool share_slots>
inline void llk_math_mul_reduce_scalar_move_product(
    const std::uint32_t i, const std::uint32_t num_products, const std::uint32_t icb0) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(icb0);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    _llk_math_mul_reduce_scalar_move_product_<share_slots>(i, num_products, tensor_shape);
}
