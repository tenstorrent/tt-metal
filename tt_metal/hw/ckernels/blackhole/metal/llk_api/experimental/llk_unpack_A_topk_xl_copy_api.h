// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK A — TopK-XL copy (single-UNPACR unpack-to-dest MOP for FP16 and FP32 paths)
 *************************************************************************/

inline void llk_unpack_topk_xl_copy_init(const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t operand_unpack_src_format = unpack_src_format[operand_id];
    const std::uint32_t operand_unpack_dst_format = unpack_dst_format[operand_id];

    ckernel::_llk_unpack_topk_xl_copy_init_(operand_unpack_src_format, operand_unpack_dst_format);
}

/*************************************************************************
 * One tile step: neg-inf clear, optional partial SETADCXX, run programmed unpack MOP.
 *************************************************************************/

inline void llk_unpack_topk_xl_copy_one_tile_unpack(
    const std::uint32_t operand, const std::uint32_t in_tile_index, const std::uint32_t elements_this_tile) {
    // Zero-element tile means "clear-only" (no source data copied).
    const std::uint32_t adc_count = (elements_this_tile == 0) ? (1024 - 1) : (elements_this_tile - 1);
    TT_SETADCXX(p_setadc::UNP_A, adc_count, 0x0);

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    const std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * in_tile_index;
    const std::uint32_t address = base_address + offset_address;

    ckernel::_llk_unpack_topk_xl_copy_(
        address, unpack_src_format[operand_id], unpack_dst_format[operand_id], elements_this_tile);
}
