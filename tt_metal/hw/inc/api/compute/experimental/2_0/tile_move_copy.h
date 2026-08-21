// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free datacopy init. Takes an LLKOperand whose data format + tile geometry are NTTPs
 * (deduced from the argument). The op forwards LLKOperand<F,S>::descriptor to the LLK as an NTTP (folds);
 * the register format is derived INSIDE the LLK. No CB id, no register format on the API surface. Legacy
 * ckernel::copy_tile_init is untouched.
 *
 * | Template | Format | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  |  | True |
 * | Template | Shape  | Tile geometry (deduced from the LLKOperand argument)         | TensorShape |  | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_tile_init(LLKOperand<Format, Shape> /*src*/) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_tile_init: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE>()));
}

// clang-format off
/**
 * Experimental id-free datacopy. Copies one tile from the L1 region described by the LLKOperand into DST.
 * Compile-time "what" = the LLKOperand NTTPs (Format + Shape, fold/DCE); runtime "where" = src.l1_address
 * (from the address seam). No CB id, no register format on the API.
 *
 * | Template | Format         | Buffer L1 data format (deduced from LLKOperand)   | DataFormat  |         | True |
 * | Template | Shape          | Tile geometry (deduced from LLKOperand)           | TensorShape |         | True |
 * | Function | src            | The source L1 operand (format+shape+address)      | LLKOperand  |         | True |
 * | Function | dst_tile_index | Tile index in the DST register                    | uint32_t    | 0 to 15 | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_tile(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_tile: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(src.l1_address)));
    MATH((llk_math_eltwise_unary_datacopy<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          UnpackToDestEn>(dst_tile_index)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
