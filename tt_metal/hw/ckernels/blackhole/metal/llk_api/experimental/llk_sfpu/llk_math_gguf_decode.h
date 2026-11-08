// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpi.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode.h"

// GGUF weight decode on the SFPU (sfpu/experimental/ckernel_sfpu_gguf_decode.h). Runs on the math or the pack thread;
// both address DEST through their own DEST_TARGET_REG_CFG_MATH_Offset.

namespace ckernel {

// BESIDE_UNPACK: whether an unpack to DEST may run beside phase 2 (the same value for init and tiles).
template <std::uint32_t FMT, bool BESIDE_UNPACK = true>
inline void llk_math_gguf_decode_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
    sfpu::sfpu_gguf_decode_init<FMT, BESIDE_UNPACK>();
}

template <std::uint32_t FMT>
inline void llk_math_gguf_decode_group(const std::uint32_t dst_index, const std::uint32_t rows) {
    sfpu::sfpu_gguf_decode_group<FMT>(dst_index, rows);
}

// PARAMS, BODY: phase 1 (scales) and phase 2 (codes); a decoded tile needs both, in that order. MXFP4 and NVFP4
// block-float codes are in the 16-bit rows that the decoded tile's rows address; in the upper half of a half-sync DEST
// those are rows 2 base + 64 dst_index, whose 32-bit rows alias the half's tiles (dest_srca_mm codes post).
template <std::uint32_t FMT, bool PARAMS = true, bool BODY = true, bool BESIDE_UNPACK = true>
inline void llk_math_gguf_decode_tile(const std::uint32_t dst_index, const std::uint32_t ti, const std::uint32_t rows) {
    const std::uint32_t r = sfpu::gguf::fp4_bfp<FMT> ? rows + get_dest_buffer_base() : rows;
    sfpu::sfpu_gguf_decode_tile<FMT, PARAMS, BODY, BESIDE_UNPACK>(dst_index, ti, r);
}

// Whether phase 2 of FMT has DEST stores that are not arbitrated against the unpackers: no copy may run beside it.
template <std::uint32_t FMT>
constexpr bool llk_math_gguf_decode_body_unarbitrated() {
    return sfpu::sfpu_gguf_decode_body_unarbitrated<FMT>();
}

// Float32 zeros in DEST tile dst_index, e.g. the accumulator of dest_srca_mm.
inline void llk_math_gguf_zero_tile(const std::uint32_t dst_index) {
    sfpu::sfpu_gguf_decode_start(dst_index, 0);
#pragma GCC unroll 8
    for (std::uint32_t i = 0; i < 32; i++) {
        sfpi::dst_reg[i] = 0.0f;
    }
    sfpu::sfpu_gguf_decode_done();
}

}  // namespace ckernel
