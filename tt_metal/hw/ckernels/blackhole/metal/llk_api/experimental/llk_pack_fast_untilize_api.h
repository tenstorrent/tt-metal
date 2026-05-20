// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_pack_fast_untilize.h"

/*************************************************************************
 * LLK PACK FAST UNTILIZE (BH)
 *************************************************************************/

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_init_with_formats(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format) {
    ckernel::_llk_pack_fast_untilize_init_<block_ct_dim, full_ct_dim>(pack_src_format, pack_dst_format);
}

template <std::uint32_t block_ct_dim>
inline void llk_pack_fast_untilize_block_at_address(
    const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim) {
    ckernel::_llk_pack_fast_untilize_block_<block_ct_dim>(address, unit_dim, prev_unit_dim);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_block_strided_at_address(
    const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim) {
    ckernel::_llk_pack_fast_untilize_block_strided_<block_ct_dim, full_ct_dim>(address, unit_dim, prev_unit_dim);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_uninit_with_src_format(const std::uint32_t pack_src_format) {
    ckernel::_llk_pack_fast_untilize_uninit_<block_ct_dim, full_ct_dim>(pack_src_format);
}
