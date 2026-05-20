// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_fast_untilize.h"

/*************************************************************************
 * LLK UNPACK FAST UNTILIZE (BH)
 *************************************************************************/

inline void llk_unpack_fast_untilize_init_with_formats(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t init_unit_dim) {
    ckernel::_llk_unpack_fast_untilize_init_<DST_ACCUM_MODE>(unpack_src_format, unpack_dst_format, init_unit_dim);
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

inline void llk_unpack_fast_untilize_uninit() { ckernel::_llk_unpack_fast_untilize_uninit_(); }
