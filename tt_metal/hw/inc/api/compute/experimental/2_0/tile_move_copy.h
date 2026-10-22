// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/dest_sync.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free copy init, Quasar. Two paths, selected by `unpack_to_dest`:
 *   * false (FPU datacopy): unpack TRISC programs an Unp0 buffer descriptor at src's base and the unpack-to-SrcA MOP;
 *     math TRISC programs the SrcA -> Dest datacopy. Joins the FPU unit to the Dest chain.
 *   * true (unpack-to-dest): unpack TRISC programs the descriptor and the UNP_DEST MOP; the unpacker writes Dest
 *     directly. Joins the UNPACK unit to the Dest chain. Full 32x32 tiles only.
 * src's base is the DFB ring base, so one init serves every front() of the ring. Re-init when another op ran in
 * between (Quasar rule). No uninit. Only 16-bit float operands (Float16_b, Float16).
 *
 * | Param Type | Name                | Description                                          | Type        | Valid Range        | Required |
 * |------------|---------------------|------------------------------------------------------|-------------|--------------------|----------|
 * | Template   | unpack_to_dest      | true: unpack straight to Dest; false: FPU datacopy   | bool        |                    | False    |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                            | bool        |                    | False    |
 * | Template   | Format / Shape      | Source L1 format + geometry (deduced)                | DataFormat/TensorShape | Float16_b, Float16 | True |
 * | Function   | src                 | Source operand (e.g. DataflowBuffer::front)          | LLKOperand  | N/A                | True     |
 */
// clang-format on
template <bool unpack_to_dest = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_init(LLKOperand<Format, Shape> src) {
    static_assert(llk_2_0::is_supported_format(Format), "copy_init (Quasar 2.0): only Float16_b / Float16.");
    static_assert(is_legal_tile_shape(Shape), "copy_init: illegal tile shape.");
    constexpr LLKMemDescriptor D = LLKOperand<Format, Shape>::descriptor;
    if constexpr (unpack_to_dest) {
        detail::dest_join<dest_dvalid_client::UNPACK>();
        UNPACK((llk_2_0::unpack_to_dest_init<D>(src.l1_address)));
    } else {
        detail::dest_leave<dest_dvalid_client::UNPACK>();
        detail::dest_join<dest_dvalid_client::FPU>();
        UNPACK((llk_2_0::unpack_A_init<D, is_fp32_dest_acc_en>(src.l1_address)));
        MATH((llk_2_0::math_datacopy_init<D, is_fp32_dest_acc_en>()));
    }
}

// clang-format off
/**
 * Id-free copy of one tile, Quasar: source tile `itile` of `src` (L1 tile src.entry_idx + itile * src.stride_tiles)
 * to Dest tile `idst`, on the path the preceding copy_init chose. Takes Dest for its unit (FPU or UNPACK) inside the
 * section the kernel opened with dest_acquire.
 *
 * | Param Type | Name           | Description                                     | Type        | Valid Range            | Required |
 * |------------|----------------|-------------------------------------------------|-------------|------------------------|----------|
 * | Template   | unpack_to_dest | Same as the preceding copy_init                 | bool        |                        | False    |
 * | Template   | Format / Shape | Source L1 format + geometry (deduced)           | DataFormat/TensorShape | N/A   | True     |
 * | Function   | src            | Source operand (its base = the copy_init's)     | LLKOperand  | N/A                    | True     |
 * | Function   | itile          | Source tile index within src                    | uint32_t    | N/A                    | True     |
 * | Function   | idst           | Dest tile index                                 | uint32_t    | < Dest tiles / section | True     |
 */
// clang-format on
template <bool unpack_to_dest = false, DataFormat Format, TensorShape Shape>
ALWI void copy_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    constexpr LLKMemDescriptor D = LLKOperand<Format, Shape>::descriptor;
    if constexpr (unpack_to_dest) {
        detail::dest_take<dest_dvalid_client::UNPACK>();
        UNPACK((llk_2_0::unpack_to_dest(src.l1_address, src.tile(itile), idst)));
    } else {
        detail::dest_take<dest_dvalid_client::FPU>();
        UNPACK((llk_2_0::unpack_A<D>(src.l1_address, src.tile(itile))));
        MATH((llk_2_0::math_datacopy(idst)));
    }
}

// clang-format off
/**
 * Id-free block copy, Quasar: copy_tile over a ct_dim x rt_dim block, source tiles start_itile + i -> Dest tiles
 * start_idst + i, i < ct_dim * rt_dim. The block must fit one Dest section (static_assert).
 *
 * | Param Type | Name                | Description                               | Type       | Valid Range                     | Required |
 * |------------|---------------------|-------------------------------------------|------------|---------------------------------|----------|
 * | Template   | ct_dim / rt_dim     | Block columns / rows (tiles)              | uint32_t   | ct_dim * rt_dim <= section size | True     |
 * | Template   | unpack_to_dest      | Same as the preceding copy_init           | bool       |                                 | False    |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                 | bool       |                                 | False    |
 * | Function   | src                 | Source operand                            | LLKOperand | N/A                             | True     |
 * | Function   | start_itile         | First source tile within src              | uint32_t   | N/A                             | True     |
 * | Function   | start_idst          | First Dest tile                           | uint32_t   | N/A                             | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool unpack_to_dest = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
ALWI void copy_block(LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst = 0) {
    static_assert(
        ct_dim * rt_dim <= detail::dest_section_tiles<is_fp32_dest_acc_en>(),
        "copy_block: block exceeds the Dest section");
    for (std::uint32_t i = 0; i < ct_dim * rt_dim; ++i) {
        copy_tile<unpack_to_dest>(src, start_itile + i, start_idst + i);
    }
}

// clang-format off
/**
 * Id-free block copy with a runtime tile count (tails), Quasar: copy_tile over `ntiles` tiles, source tiles
 * start_itile + i -> Dest tiles start_idst + i.
 *
 * | Param Type | Name                | Description                               | Type       | Valid Range                          | Required |
 * |------------|---------------------|-------------------------------------------|------------|--------------------------------------|----------|
 * | Template   | unpack_to_dest      | Same as the preceding copy_init           | bool       |                                      | False    |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                 | bool       |                                      | False    |
 * | Function   | src                 | Source operand                            | LLKOperand | N/A                                  | True     |
 * | Function   | start_itile         | First source tile within src              | uint32_t   | N/A                                  | True     |
 * | Function   | start_idst          | First Dest tile                           | uint32_t   | N/A                                  | True     |
 * | Function   | ntiles              | Number of tiles                           | uint32_t   | start_idst + ntiles <= section size  | True     |
 */
// clang-format on
template <bool unpack_to_dest = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst, std::uint32_t ntiles) {
    LLK_ASSERT(
        start_idst + ntiles <= detail::dest_section_tiles<is_fp32_dest_acc_en>(), "copy_block: block exceeds Dest");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        copy_tile<unpack_to_dest>(src, start_itile + i, start_idst + i);
    }
}

}  // namespace experimental
}  // namespace ckernel

#else  // Blackhole

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "data_format_derive.h"  // ckernel::infer_unpack_dst_format -- register-format derivation (BH)

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#endif

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free datacopy init. Takes an LLKOperand whose data format + tile geometry are NTTPs (deduced from the
 * argument); the register format is derived inside the LLK. Reprograms the unpacker and math datacopy for the
 * operand's format without touching the packer dst-offset registers (those come from
 * compute_kernel_hw_startup). Blackhole only.
 *
 * fp8 (Fp8_e4m3 / Lf8 e5m2) is supported: the format-driven Src zero-substitution flag is programmed so fp8
 * zeros read back correctly, while other formats (e.g. bf16 -0.0 / 16b-int) keep their residual value.
 *
 * transpose / transpose_within_16x16_face support copy-with-transpose; both default off (0), so the default
 * copy path is unaffected.
 *
 * Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Template | is_fp32_dest_acc_en         | fp32 dest-accumulate mode                                    | bool        |                                                                     | False |
 * | Template | Format                      | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  |                                                                     | True  |
 * | Template | Shape                       | Tile geometry (deduced from the LLKOperand argument)         | TensorShape |                                                                     | True  |
 * | Function | transpose                   | Flag to perform transpose on SrcA                           | uint32_t    | Any positive value will indicate transpose is set                   | False |
 * | Function | transpose_within_16x16_face | Flag to perform transpose within 16x16 face                 | uint32_t    | Any positive value will indicate transpose within 16x16 face is set | False |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_init(
    LLKOperand<Format, Shape> /*src*/, std::uint32_t transpose = 0, std::uint32_t transpose_within_16x16_face = 0) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_init: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(transpose, transpose_within_16x16_face)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          is_fp32_dest_acc_en,
          BroadcastType::NONE>()));
    // Src zero-substitution flag, chosen by the derived register format (id-free equivalent of legacy
    // copy_init's ckernel::math::_configure_copy_zero_flag_state_(get_operand_dst_format(cbid))): preserve
    // bf16 -0.0 / 16b-int residuals, but flush fp8 (e4m3/e5m2) whose zero widens to a nonzero SrcA residual
    // and must read back as 0. MATH-only config; records no format-reconfig diff. The fn masks to 0x1F.
    MATH((ckernel::math::_configure_copy_zero_flag_state_(
        static_cast<std::uint32_t>(ckernel::infer_unpack_dst_format(Format, is_fp32_dest_acc_en)))));
}

// clang-format off
/**
 * Id-free datacopy. Copies tile `itile` from the L1 buffer described by the LLKOperand into DST.
 * `src.l1_address` is the buffer base; the unpack address is `tile_address(src, itile)`. Blackhole only.
 *
 * Requires copy_init to have run, and the DST register to be acquired (tile_regs_acquire) before this call.
 * Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                          | bool        |         | False |
 * | Template | Format         | Buffer L1 data format (deduced from LLKOperand)   | DataFormat  |         | True |
 * | Template | Shape          | Tile geometry (deduced from LLKOperand)           | TensorShape |         | True |
 * | Function | src            | The source L1 operand (format+shape+buffer base)  | LLKOperand  |         | True |
 * | Function | itile          | Index of the tile within `src`, relative to its base | uint32_t | N/A     | True |
 * | Function | dst_tile_index | Tile index in the DST register                    | uint32_t    | Must be less than the acquired size of DST REG | True |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t dst_tile_index) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_tile: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(detail::tile_address(src, itile))));
    MATH((llk_math_eltwise_unary_datacopy<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          is_fp32_dest_acc_en,
          BroadcastType::NONE,
          UnpackToDestEn>(dst_tile_index)));
}

// clang-format off
/**
 * Id-free block datacopy. Copies `ntiles` consecutive tiles from the L1 buffer described by the LLKOperand
 * into consecutive DST register slots (tile start_in_tile_index+i lands in DST slot start_dst_tile_index+i).
 * Loop form of copy_tile. Blackhole only.
 *
 * Requires the same init as copy_tile (copy_init). Sub-32-row (partial-height) block-float tiles are not
 * supported (compile-time rejected).
 *
 * | Template | is_fp32_dest_acc_en  | fp32 dest-accumulate mode                                    | bool        |         | False |
 * | Template | Format               | Buffer L1 data format (deduced from LLKOperand)             | DataFormat  |         | True |
 * | Template | Shape                | Tile geometry (deduced from LLKOperand)                    | TensorShape |         | True |
 * | Function | src                  | The source L1 operand (format+shape+base address)          | LLKOperand  |         | True |
 * | Function | start_in_tile_index  | Index of the first source tile, relative to src base        | uint32_t    | N/A     | True |
 * | Function | start_dst_tile_index | Index of the first destination tile in the DST register     | uint32_t    | Must be less than the acquired size of DST REG | True |
 * | Function | ntiles               | Number of consecutive tiles to copy                         | uint32_t    | start_dst_tile_index + ntiles <= acquired size of DST REG | True |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_block(
    LLKOperand<Format, Shape> src,
    std::uint32_t start_in_tile_index,
    std::uint32_t start_dst_tile_index,
    std::uint32_t ntiles) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_block: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        copy_tile<is_fp32_dest_acc_en>(src, start_in_tile_index + i, start_dst_tile_index + i);
    }
}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
