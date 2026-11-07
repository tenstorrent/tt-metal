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

template <std::uint32_t FMT>
inline void llk_math_gguf_decode_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
    sfpu::sfpu_gguf_decode_init<FMT>();
}

template <std::uint32_t FMT>
inline void llk_math_gguf_decode_group(const std::uint32_t dst_index, const std::uint32_t rows) {
    sfpu::sfpu_gguf_decode_group<FMT>(dst_index, rows);
}

// PARAMS, BODY: phase 1 (scales) and phase 2 (codes); a decoded tile needs both, in that order.
template <std::uint32_t FMT, bool PARAMS = true, bool BODY = true>
inline void llk_math_gguf_decode_tile(const std::uint32_t dst_index, const std::uint32_t ti, const std::uint32_t rows) {
    sfpu::sfpu_gguf_decode_tile<FMT, PARAMS, BODY>(dst_index, ti, rows);
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
