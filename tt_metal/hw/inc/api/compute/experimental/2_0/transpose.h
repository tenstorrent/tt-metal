// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#include "llk_math_transpose_dest_api.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#include "llk_unpack_common_api.h"  // llk_unpack_set_srcb_dummy_valid
#endif

// =====================================================================================================
// Id-free (2.0) transpose: transposes one input tile into DST. Takes a single input LLKOperand (format +
// tile geometry as NTTPs). The caller picks the path: transpose_src_to_dest_* (unpack to SrcA with face and
// within-face transpose, then an FPU A2D datacopy; 8-bit integer formats use the int-FPU reconstruct) or
// transpose_direct_to_dest_* (unpack straight into DST with face transpose, then a 32-bit transpose_dest of
// each face in DST). Blackhole only.
// =====================================================================================================

namespace ckernel {
namespace experimental {

namespace detail {

// Legality per path, from the BH LLK: SrcA is 19 bits wide, so a 32-bit register format (Float32 at fp32 Dest,
// Int32, UInt32) cannot take the Src path; the LLK unpacks to Dest only for a 32-bit input (should_unpack_to_dest)
// and the direct path's transpose_dest is the 32-bit variant, so every other format must take the Src path. Both
// paths transpose faces in the unpacker, which takes only 16-row faces and 1 or 4 faces (_llk_unpack_A_init_).
template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
constexpr void transpose_check() {
    static_assert(
        Shape.face_r_dim == 16 && Shape.face_c_dim == 16 &&
            (Shape.total_num_faces() == 1 || Shape.total_num_faces() == 4),
        "transpose: the unpacker transposes only 16x16 faces, 1 or 4 per tile (32x32 or 16x16 tiles).");
    static_assert(
        direct || !is_unpack_to_dest<Format, is_fp32_dest_acc_en>(),
        "transpose_src_to_dest: a 32-bit register format (Float32 at fp32 Dest, Int32, UInt32) does not fit the "
        "19-bit SrcA; use transpose_direct_to_dest_*.");
    static_assert(
        !direct || is_unpack_to_dest<Format, is_fp32_dest_acc_en>(),
        "transpose_direct_to_dest: unpack-to-Dest takes only a 32-bit register format (Float32 at fp32 Dest, Int32, "
        "UInt32); use transpose_src_to_dest_*.");
}

template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
ALWI void transpose_init() {
    transpose_check<direct, is_fp32_dest_acc_en, Format, Shape>();
    if constexpr (direct) {
        // Faces are transposed by the unpacker; within-face transpose is done in DST by transpose_dest.
        UNPACK((llk_unpack_A_init<
                LLKOperand<Format, Shape>::descriptor,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE,
                true /*unpack_to_dest*/>(true /*transpose_of_faces*/, false /*within_face_16x16_transpose*/)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              is_fp32_dest_acc_en,
              BroadcastType::NONE>()));
        MATH((llk_math_transpose_dest_init<false /*transpose_of_faces*/, true /*is_32bit*/>()));
    } else {
        // Low-nibble compare intentionally matches both signed Int8 (14) and unsigned UInt8 (30 -> low nibble
        // 0xE): both 8-bit integer formats need the int-FPU (ELWADD) A2D reconstruct path.
        constexpr bool is_8bit_int =
            (static_cast<std::uint8_t>(Format) & 0xf) == static_cast<std::uint8_t>(DataFormat::Int8);
        UNPACK((llk_unpack_A_init<
                LLKOperand<Format, Shape>::descriptor,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                true /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE>(true /*transpose_of_faces*/, true /*within_face_16x16_transpose*/)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              is_fp32_dest_acc_en,
              BroadcastType::NONE,
              is_8bit_int /*is_int_fpu_en*/>()));
    }
}

template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
ALWI void transpose_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    transpose_check<direct, is_fp32_dest_acc_en, Format, Shape>();
    const std::uint32_t addr = tile_address(src, itile);
    if constexpr (direct) {
        UNPACK((llk_unpack_A<
                LLKOperand<Format, Shape>::descriptor,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE,
                true /*unpack_to_dest*/>(addr)));
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
        MATH((llk_math_eltwise_unary_datacopy<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              is_fp32_dest_acc_en,
              BroadcastType::NONE,
              true /*unpack_to_dest*/>(idst)));
        MATH((llk_math_transpose_dest<false /*transpose_of_faces*/, true /*is_32bit*/>(idst)));
    } else {
        UNPACK((llk_unpack_A<
                LLKOperand<Format, Shape>::descriptor,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                false /*acc_to_dest*/>(addr)));
        MATH((llk_math_eltwise_unary_datacopy<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              is_fp32_dest_acc_en,
              BroadcastType::NONE>(idst)));
    }
}

}  // namespace detail

// clang-format off
/**
 * Init for transpose_src_to_dest_tile / transpose_src_to_dest_block, Src path: the unpacker transposes faces and
 * within each face into SrcA, then an FPU A2D datacopy moves the tile into DST. 8-bit integer formats (Int8/UInt8)
 * take the int-FPU (ELWADD) A2D reconstruct. Takes an LLKOperand whose data format + tile geometry are NTTPs
 * (deduced from the argument). compute_kernel_hw_startup must already have run once.
 *
 * Any format whose register format is not 32-bit. A 32-bit register format (Float32 at fp32 Dest, Int32, UInt32)
 * does not fit the 19-bit SrcA and is rejected at compile time: use transpose_direct_to_dest_init. 32x32 or 16x16
 * tiles only (16-row faces, 1 or 4 faces).
 *
 * | Param Type | Name                | Description                                                  | Type        | Valid Range                | Required |
 * |------------|---------------------|--------------------------------------------------------------|-------------|----------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                    | bool        | N/A                        | False    |
 * | Template   | Format              | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  | Register format not 32-bit | True     |
 * | Template   | Shape               | Tile geometry (deduced from the LLKOperand argument)         | TensorShape | 32x32, 16x16               | True     |
 * | Function   | src                 | The source L1 operand (format + shape; address unused here)  | LLKOperand  | N/A                        | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void transpose_src_to_dest_init(LLKOperand<Format, Shape> /*src*/) {
    detail::transpose_init<false, is_fp32_dest_acc_en, Format, Shape>();
}

// clang-format off
/**
 * Init for transpose_direct_to_dest_tile / transpose_direct_to_dest_block, direct path: the unpacker writes the
 * tile straight into DST (unpack-to-dest, all 32 bits) with its faces transposed, then math transposes each face
 * in DST (32-bit transpose_dest). Takes an LLKOperand whose data format + tile geometry are NTTPs (deduced from the
 * argument). compute_kernel_hw_startup must already have run once.
 *
 * Only a 32-bit register format: Float32 at fp32 Dest, Int32, UInt32 (compile-time checked). For any other format
 * use transpose_src_to_dest_init. 32x32 or 16x16 tiles only (16-row faces, 1 or 4 faces).
 *
 * | Param Type | Name                | Description                                                  | Type        | Valid Range                        | Required |
 * |------------|---------------------|--------------------------------------------------------------|-------------|------------------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                    | bool        | N/A                                | False    |
 * | Template   | Format              | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  | Float32 (fp32 Dest), Int32, UInt32 | True     |
 * | Template   | Shape               | Tile geometry (deduced from the LLKOperand argument)         | TensorShape | 32x32, 16x16                       | True     |
 * | Function   | src                 | The source L1 operand (format + shape; address unused here)  | LLKOperand  | N/A                                | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void transpose_direct_to_dest_init(LLKOperand<Format, Shape> /*src*/) {
    detail::transpose_init<true, is_fp32_dest_acc_en, Format, Shape>();
}

// clang-format off
/**
 * Src path: performs a 32x32 transpose *B[w,h] = A[h,w]* on tile `itile` of the L1 region described by the
 * LLKOperand, through SrcA, and writes the result to DST[idst]. DST must be in the acquired state
 * (tile_regs_acquire). Pair with transpose_src_to_dest_init. Blocking; compute engine only.
 *
 * | Param Type | Name                | Description                                                   | Type        | Valid Range                                    | Required |
 * |------------|---------------------|---------------------------------------------------------------|-------------|------------------------------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                     | bool        | N/A                                            | False    |
 * | Template   | Format              | Buffer L1 data format (deduced from the LLKOperand argument)  | DataFormat  | Register format not 32-bit                     | True     |
 * | Template   | Shape               | Tile geometry (deduced from the LLKOperand argument)          | TensorShape | 32x32, 16x16                                   | True     |
 * | Function   | src                 | The source L1 operand (format + shape + base address)         | LLKOperand  | N/A                                            | True     |
 * | Function   | itile               | Index of the tile A within `src`, relative to its base address| uint32_t    | N/A                                            | True     |
 * | Function   | idst                | Index of the tile in DST REG for the result B                 | uint32_t    | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void transpose_src_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    detail::transpose_tile<false, is_fp32_dest_acc_en>(src, itile, idst);
}

// clang-format off
/**
 * Direct path: performs a 32x32 transpose *B[w,h] = A[h,w]* on tile `itile` of the L1 region described by the
 * LLKOperand, unpacked straight into DST[idst] and transposed there. DST must be in the acquired state
 * (tile_regs_acquire). Pair with transpose_direct_to_dest_init. Blocking; compute engine only.
 *
 * | Param Type | Name                | Description                                                   | Type        | Valid Range                                    | Required |
 * |------------|---------------------|---------------------------------------------------------------|-------------|------------------------------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                     | bool        | N/A                                            | False    |
 * | Template   | Format              | Buffer L1 data format (deduced from the LLKOperand argument)  | DataFormat  | Float32 (fp32 Dest), Int32, UInt32             | True     |
 * | Template   | Shape               | Tile geometry (deduced from the LLKOperand argument)          | TensorShape | 32x32, 16x16                                   | True     |
 * | Function   | src                 | The source L1 operand (format + shape + base address)         | LLKOperand  | N/A                                            | True     |
 * | Function   | itile               | Index of the tile A within `src`, relative to its base address| uint32_t    | N/A                                            | True     |
 * | Function   | idst                | Index of the tile in DST REG for the result B                 | uint32_t    | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void transpose_direct_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    detail::transpose_tile<true, is_fp32_dest_acc_en>(src, itile, idst);
}

// clang-format off
/**
 * Src path, ct_dim x rt_dim block: tile start_itile + i of `src` is transposed into DST slot start_idst + i. Loop form
 * of transpose_src_to_dest_tile. Requires transpose_src_to_dest_init and an acquired DST. No tails: the block divides
 * the full dim, so every block op covers the same tile count.
 *
 * | Param Type | Name                | Description                                                       | Type       | Valid Range                          | Required |
 * |------------|---------------------|-------------------------------------------------------------------|------------|--------------------------------------|----------|
 * | Template   | ct_dim / rt_dim     | Block columns / rows (tiles)                                      | uint32_t   | ct_dim * rt_dim <= acquired DST size | True     |
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                         | bool       | N/A                                  | False    |
 * | Function   | src                 | The source L1 operand (format + shape + base address)             | LLKOperand | Register format not 32-bit           | True     |
 * | Function   | start_itile         | Index of the first tile A within `src`, relative to its base addr | uint32_t   | N/A                                  | True     |
 * | Function   | start_idst          | Index of the first destination tile in DST REG                    | uint32_t   | N/A                                  | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
ALWI void transpose_src_to_dest_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst = 0) {
    for (std::uint32_t i = 0; i < ct_dim * rt_dim; ++i) {
        detail::transpose_tile<false, is_fp32_dest_acc_en>(src, start_itile + i, start_idst + i);
    }
}

// clang-format off
/**
 * Direct path, ct_dim x rt_dim block: tile start_itile + i of `src` is unpacked into DST slot start_idst + i and
 * transposed there. Loop form of transpose_direct_to_dest_tile. Requires transpose_direct_to_dest_init and an
 * acquired DST. No tails: the block divides the full dim, so every block op covers the same tile count.
 *
 * | Param Type | Name                | Description                                                       | Type       | Valid Range                          | Required |
 * |------------|---------------------|-------------------------------------------------------------------|------------|--------------------------------------|----------|
 * | Template   | ct_dim / rt_dim     | Block columns / rows (tiles)                                      | uint32_t   | ct_dim * rt_dim <= acquired DST size | True     |
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                         | bool       | N/A                                  | False    |
 * | Function   | src                 | The source L1 operand (format + shape + base address)             | LLKOperand | Float32 (fp32 Dest), Int32, UInt32   | True     |
 * | Function   | start_itile         | Index of the first tile A within `src`, relative to its base addr | uint32_t   | N/A                                  | True     |
 * | Function   | start_idst          | Index of the first destination tile in DST REG                    | uint32_t   | N/A                                  | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
ALWI void transpose_direct_to_dest_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst = 0) {
    for (std::uint32_t i = 0; i < ct_dim * rt_dim; ++i) {
        detail::transpose_tile<true, is_fp32_dest_acc_en>(src, start_itile + i, start_idst + i);
    }
}

}  // namespace experimental
}  // namespace ckernel
