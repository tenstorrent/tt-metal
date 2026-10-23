// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_AB_scalar_block.h"
#include "llk_unpack_common_api.h"

#include "sanitizer/api.h"

namespace ckernel {

inline void llk_unpack_AB_scalar_block_init(const std::uint32_t operand_a, const std::uint32_t operand_b) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_a_id = get_operand_id(operand_a);
    const std::uint32_t operand_b_id = get_operand_id(operand_b);
    _llk_unpack_AB_scalar_block_init_(get_operand_tensor_shape(operand_a_id), get_operand_tensor_shape(operand_b_id));
}

inline void llk_unpack_AB_scalar_block(
    const std::uint32_t operand_a,
    const std::uint32_t operand_b,
    const std::uint32_t tile_a,
    const std::uint32_t tile_b,
    const std::uint32_t block_size) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_a_id = get_operand_id(operand_a);
    const std::uint32_t operand_b_id = get_operand_id(operand_b);
    const std::uint32_t address_a = get_local_cb_interface(operand_a_id).fifo_rd_ptr - 1 +
                                    get_local_cb_interface(operand_a_id).fifo_page_size * tile_a;
    const std::uint32_t address_b = get_local_cb_interface(operand_b_id).fifo_rd_ptr - 1 +
                                    get_local_cb_interface(operand_b_id).fifo_page_size * tile_b;
    _llk_unpack_AB_scalar_block_(address_a, address_b, block_size);
}

}  // namespace ckernel
