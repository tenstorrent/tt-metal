// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"
#include "experimental/llk_unpack_fast_untilize.h"

/*************************************************************************
 * LLK UNPACK FAST UNTILIZE (BH)
 *************************************************************************/

inline void llk_unpack_fast_untilize_init_with_formats(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t init_unit_dim) {
    ckernel::_llk_unpack_fast_untilize_init_<DST_ACCUM_MODE>(unpack_src_format, unpack_dst_format, init_unit_dim);
}

inline bool llk_unpack_fast_untilize_is_bfp_b_input_format(const std::uint32_t format) {
    return format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b);
}

inline bool llk_unpack_fast_untilize_is_bfp_b_input(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    return llk_unpack_fast_untilize_is_bfp_b_input_format(unpack_src_format[operand_id]);
}

inline void llk_unpack_fast_untilize_init(const std::uint32_t operand, const std::uint32_t dense_init_unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t init_unit_dim =
        llk_unpack_fast_untilize_is_bfp_b_input_format(unpack_src_format[operand_id]) ? 1 : dense_init_unit_dim;
    ckernel::_llk_unpack_fast_untilize_init_<DST_ACCUM_MODE>(
        unpack_src_format[operand_id], unpack_dst_format[operand_id], init_unit_dim);
}

inline void llk_unpack_fast_untilize_reinit_unit_dim(const std::uint32_t unit_dim) {
    ckernel::_llk_unpack_fast_untilize_reinit_unit_dim_<DST_ACCUM_MODE>(unit_dim);
}

inline void llk_unpack_fast_untilize_block_at_address(const std::uint32_t address, const std::uint32_t unit_dim) {
    ckernel::_llk_unpack_fast_untilize_block_(address, unit_dim);
}

inline void llk_unpack_fast_untilize_bfp_block_at_address(
    const std::uint32_t address, const std::uint32_t tile_stride_16B, const std::uint32_t unit_dim) {
    ckernel::_llk_unpack_fast_untilize_bfp_block_(address, tile_stride_16B, unit_dim);
}

inline std::uint32_t llk_unpack_fast_untilize_tile_address(
    const std::uint32_t operand_id, const std::uint32_t tile_index) {
    const auto& cb_interface = get_local_cb_interface(operand_id);
    return cb_interface.fifo_rd_ptr + cb_interface.fifo_page_size * tile_index - 1;
}

inline void llk_unpack_fast_untilize_block(
    const std::uint32_t operand, const std::uint32_t tile_index, const std::uint32_t unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t address = llk_unpack_fast_untilize_tile_address(operand_id, tile_index);
    if (llk_unpack_fast_untilize_is_bfp_b_input_format(unpack_src_format[operand_id])) {
        llk_unpack_fast_untilize_bfp_block_at_address(
            address, get_local_cb_interface(operand_id).fifo_page_size, unit_dim);
    } else {
        llk_unpack_fast_untilize_block_at_address(address, unit_dim);
    }
}

inline void llk_unpack_fast_untilize_block(
    const std::uint32_t operand,
    const std::uint32_t tile_index,
    const std::uint32_t unit_dim,
    std::uint32_t& prev_unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t address = llk_unpack_fast_untilize_tile_address(operand_id, tile_index);
    if (llk_unpack_fast_untilize_is_bfp_b_input_format(unpack_src_format[operand_id])) {
        llk_unpack_fast_untilize_bfp_block_at_address(
            address, get_local_cb_interface(operand_id).fifo_page_size, unit_dim);
        return;
    }

    if (unit_dim != prev_unit_dim) {
        llk_unpack_fast_untilize_reinit_unit_dim(unit_dim);
        prev_unit_dim = unit_dim;
    }
    llk_unpack_fast_untilize_block_at_address(address, unit_dim);
}

inline void llk_unpack_fast_untilize_restore_unit_dim(
    const std::uint32_t operand, const std::uint32_t init_unit_dim, const std::uint32_t prev_unit_dim) {
    if (!llk_unpack_fast_untilize_is_bfp_b_input(operand) && prev_unit_dim != init_unit_dim) {
        llk_unpack_fast_untilize_reinit_unit_dim(init_unit_dim);
    }
}

inline void llk_unpack_fast_untilize_uninit() { ckernel::_llk_unpack_fast_untilize_uninit_(); }
