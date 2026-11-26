// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common_api.h"
#include "experimental/llk_math_bcast_dest_reuse.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK ELTWISE BINARY BCAST DEST REUSE
 *************************************************************************/

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    MathFidelity math_fidelity>
inline void llk_math_bcast_dest_reuse_init(const std::uint32_t operand_A, const std::uint32_t acc_to_dest = 0) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_bcast_dest_reuse_init_<eltwise_binary_type, bcast_type, num_tiles, math_fidelity>(
        tensor_shape, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool clear_dest = false>
inline void llk_math_bcast_dest_reuse(
    const std::uint32_t operand_A, const std::uint32_t src_index, const std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_bcast_dest_reuse_<
        eltwise_binary_type,
        bcast_type,
        num_tiles,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        clear_dest>(src_index, dst_index, tensor_shape);
}
