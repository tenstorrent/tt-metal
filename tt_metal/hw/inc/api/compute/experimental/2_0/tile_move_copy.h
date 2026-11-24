// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/internal/op_hooks.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

namespace detail {

// The caller picks the path: Src (unpack to SrcA, then an FPU datacopy into Dest) or direct (unpack straight into
// Dest). Legality per path, from llk_2_0: SrcA takes the 16-bit float formats only; the unpack-to-dest path is wired
// for Float32 at a 32-bit Dest (all 32 bits land in Dest).
template <bool direct, DataFormat Format>
constexpr void copy_check() {
    static_assert(
        direct || llk_2_0::is_supported_format(Format),
        "copy_src_to_dest (Quasar 2.0): SrcA takes only Float16_b / Float16. For Float32 at a 32-bit Dest use "
        "copy_direct_to_dest_*.");
    static_assert(
        !direct || (DST_ACCUM_MODE && Format == DataFormat::Float32),
        "copy_direct_to_dest (Quasar 2.0): unpack-to-dest takes only Float32 at a 32-bit Dest (DST_ACCUM_MODE). For "
        "Float16_b / Float16 use copy_src_to_dest_*.");
}

template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
inline void copy_init(LLKOperand<Format, Shape> src) {
    copy_check<direct, Format>();
    static_assert(is_legal_tile_shape(Shape), "copy init: illegal tile shape.");
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "copy init: Dest mode is kernel-wide (DST_ACCUM_MODE)");
    constexpr LLKMemDescriptor D = LLKOperand<Format, Shape>::descriptor;
    if constexpr (direct) {
        UNPACK((llk_2_0::unpack_to_dest_init<D>(src.l1_address)));  // no math side: the unpacker writes all 32 bits
    } else {
        UNPACK((llk_2_0::unpack_A_init<D, is_fp32_dest_acc_en>(src.l1_address)));
        MATH((llk_2_0::math_datacopy_init<D, is_fp32_dest_acc_en>()));
    }
}

// ntiles tiles: source tiles start_itile + i -> Dest tiles start_idst + i. The direct path borrows the Dest section
// from MATH for the call. It calls the LLK (_llk_tile_regs_borrow_) directly: tile_regs_borrow<MATH, UNPACK> compiles
// away unless the JIT flag UnpackToDestEn is set, and that flag comes only from a DFB's unpack_modes; a Float32
// operand with no such DFB entry (e.g. a local tensor) still needs the borrow.
template <bool direct, bool is_fp32_dest_acc_en, std::uint32_t ntiles, DataFormat Format, TensorShape Shape>
inline void copy_tiles(LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst) {
    copy_check<direct, Format>();
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "copy: Dest mode is kernel-wide (DST_ACCUM_MODE)");
    static_assert(ntiles <= dest_section_tiles<is_fp32_dest_acc_en>(), "copy: block exceeds the Dest section");
    LLK_ASSERT(start_idst + ntiles <= dest_section_tiles<is_fp32_dest_acc_en>(), "copy: block exceeds Dest");
    if constexpr (direct) {
        _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
        for (std::uint32_t i = 0; i < ntiles; ++i) {
            UNPACK((llk_2_0::unpack_to_dest(src.l1_address, src.tile(start_itile + i), start_idst + i)));
        }
        _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
    } else {
        for (std::uint32_t i = 0; i < ntiles; ++i) {
            UNPACK(
                (llk_2_0::unpack_A<LLKOperand<Format, Shape>::descriptor>(src.l1_address, src.tile(start_itile + i))));
            MATH((llk_2_0::math_datacopy(start_idst + i)));
        }
    }
}

}  // namespace detail

// clang-format off
/**
 * Id-free copy init, Quasar, Src path: unpack to SrcA, then an FPU datacopy into Dest. Unpack TRISC programs an Unp0
 * buffer descriptor at src's base and the unpack-to-SrcA MOP; math TRISC programs the SrcA -> Dest datacopy.
 * src's base is the DFB ring base, so one init serves every front() of the ring. Re-init when another op ran in
 * between (Quasar rule). No uninit. Float16_b / Float16 at either Dest width (static_assert); for Float32 use
 * copy_direct_to_dest_init.
 *
 * | Param Type | Name                | Description                                  | Type                   | Valid Range        | Required |
 * |------------|---------------------|----------------------------------------------|------------------------|--------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                    | bool                   | DST_ACCUM_MODE     | False    |
 * | Template   | Format / Shape      | Source L1 format + geometry (deduced)        | DataFormat/TensorShape | Float16_b, Float16 | True     |
 * | Function   | src                 | Source operand (e.g. DataflowBuffer::front)  | LLKOperand             | N/A                | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
inline void copy_src_to_dest_init(LLKOperand<Format, Shape> src) {
    detail::copy_init<false, is_fp32_dest_acc_en>(src);
}

// clang-format off
/**
 * Id-free copy init, Quasar, direct path: the unpacker writes straight into Dest (UNP_DEST MOP), all 32 bits; no math
 * side. Float32 at a 32-bit Dest only (static_assert); full 32x32 tiles only. Re-init when another op ran in between
 * (Quasar rule). No uninit. For Float16_b / Float16 use copy_src_to_dest_init.
 *
 * | Param Type | Name                | Description                                  | Type                   | Valid Range              | Required |
 * |------------|---------------------|----------------------------------------------|------------------------|--------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                    | bool                   | DST_ACCUM_MODE           | False    |
 * | Template   | Format / Shape      | Source L1 format + geometry (deduced)        | DataFormat/TensorShape | Float32 (32-bit Dest), 32x32 | True |
 * | Function   | src                 | Source operand (e.g. DataflowBuffer::front)  | LLKOperand             | N/A                      | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
inline void copy_direct_to_dest_init(LLKOperand<Format, Shape> src) {
    detail::copy_init<true, is_fp32_dest_acc_en>(src);
}

// clang-format off
/**
 * Id-free copy of one tile, Quasar, Src path: source tile `itile` of `src` (L1 tile src.entry_idx + itile *
 * src.stride_tiles) through SrcA to Dest tile `idst`. Pair with copy_src_to_dest_init. Call between tile_regs_acquire
 * and tile_regs_commit.
 *
 * | Param Type | Name                | Description                                 | Type                   | Valid Range            | Required |
 * |------------|---------------------|---------------------------------------------|------------------------|------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                   | bool                   | DST_ACCUM_MODE         | False    |
 * | Template   | Format / Shape      | Source L1 format + geometry (deduced)       | DataFormat/TensorShape | Float16_b, Float16     | True     |
 * | Function   | src                 | Source operand (its base = the init's)      | LLKOperand             | N/A                    | True     |
 * | Function   | itile               | Source tile index within src                | uint32_t               | N/A                    | True     |
 * | Function   | idst                | Dest tile index                             | uint32_t               | < Dest tiles / section | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
inline void copy_src_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    detail::copy_tiles<false, is_fp32_dest_acc_en, 1>(src, itile, idst);
}

// clang-format off
/**
 * Id-free copy of one tile, Quasar, direct path: source tile `itile` of `src` unpacked straight into Dest tile
 * `idst`; UNPACK borrows the Dest section from MATH for the call. Pair with copy_direct_to_dest_init. Call between
 * tile_regs_acquire and tile_regs_commit.
 *
 * | Param Type | Name                | Description                                 | Type                   | Valid Range            | Required |
 * |------------|---------------------|---------------------------------------------|------------------------|------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                   | bool                   | DST_ACCUM_MODE         | False    |
 * | Template   | Format / Shape      | Source L1 format + geometry (deduced)       | DataFormat/TensorShape | Float32 (32-bit Dest)  | True     |
 * | Function   | src                 | Source operand (its base = the init's)      | LLKOperand             | N/A                    | True     |
 * | Function   | itile               | Source tile index within src                | uint32_t               | N/A                    | True     |
 * | Function   | idst                | Dest tile index                             | uint32_t               | < Dest tiles / section | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
inline void copy_direct_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    detail::copy_tiles<true, is_fp32_dest_acc_en, 1>(src, itile, idst);
}

// clang-format off
/**
 * Id-free block copy, Quasar, Src path: a ct_dim x rt_dim block, source tiles start_itile + i -> Dest tiles
 * start_idst + i, through SrcA. The block must fit one Dest section (static_assert). No tails: every block op
 * between two inits must cover the same tile count; for leftover tiles re-init, then run a second, shorter
 * instantiation.
 *
 * | Param Type | Name                | Description                               | Type       | Valid Range                     | Required |
 * |------------|---------------------|-------------------------------------------|------------|---------------------------------|----------|
 * | Template   | ct_dim / rt_dim     | Block columns / rows (tiles)              | uint32_t   | ct_dim * rt_dim <= section size | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                 | bool       | DST_ACCUM_MODE                  | False    |
 * | Function   | src                 | Source operand                            | LLKOperand | Float16_b, Float16              | True     |
 * | Function   | start_itile         | First source tile within src              | uint32_t   | N/A                             | True     |
 * | Function   | start_idst          | First Dest tile                           | uint32_t   | N/A                             | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
inline void copy_src_to_dest_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst = 0) {
    detail::copy_tiles<false, is_fp32_dest_acc_en, ct_dim * rt_dim>(src, start_itile, start_idst);
}

// clang-format off
/**
 * Id-free block copy, Quasar, direct path: a ct_dim x rt_dim block, source tiles start_itile + i unpacked straight
 * into Dest tiles start_idst + i, borrowing Dest once for the block. The block must fit one Dest section
 * (static_assert). No tails: for leftover tiles re-init, then run a second, shorter instantiation.
 *
 * | Param Type | Name                | Description                               | Type       | Valid Range                     | Required |
 * |------------|---------------------|-------------------------------------------|------------|---------------------------------|----------|
 * | Template   | ct_dim / rt_dim     | Block columns / rows (tiles)              | uint32_t   | ct_dim * rt_dim <= section size | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                 | bool       | DST_ACCUM_MODE                  | False    |
 * | Function   | src                 | Source operand                            | LLKOperand | Float32 (32-bit Dest)           | True     |
 * | Function   | start_itile         | First source tile within src              | uint32_t   | N/A                             | True     |
 * | Function   | start_idst          | First Dest tile                           | uint32_t   | N/A                             | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
inline void copy_direct_to_dest_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst = 0) {
    detail::copy_tiles<true, is_fp32_dest_acc_en, ct_dim * rt_dim>(src, start_itile, start_idst);
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

namespace detail {

// The caller picks the path: Src (unpack to SrcA, then an FPU A2D datacopy into Dest) or direct (unpack straight into
// Dest). Legality per path, from the LLK: SrcA is 19 bits wide, so a 32-bit register format (Float32 at fp32 Dest,
// Int32, UInt32) cannot take the Src path (Int32 has no SrcA conversion at all, Float32 -> Tf32 would drop mantissa
// bits); the LLK unpacks to Dest only for a 32-bit input (should_unpack_to_dest), so every other format must take
// the Src path.
template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
constexpr void copy_check() {
    static_assert(
        is_legal_tile_shape(Shape), "copy: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    static_assert(
        direct || !is_unpack_to_dest<Format, is_fp32_dest_acc_en>(),
        "copy_src_to_dest: a 32-bit register format (Float32 at fp32 Dest, Int32, UInt32) does not fit the 19-bit "
        "SrcA; use copy_direct_to_dest_*.");
    static_assert(
        !direct || is_unpack_to_dest<Format, is_fp32_dest_acc_en>(),
        "copy_direct_to_dest: unpack-to-Dest takes only a 32-bit register format (Float32 at fp32 Dest, Int32, "
        "UInt32); use copy_src_to_dest_*.");
}

template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
ALWI void copy_init(std::uint32_t transpose, std::uint32_t transpose_within_16x16_face) {
    copy_check<direct, is_fp32_dest_acc_en, Format, Shape>();
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            direct>(transpose, transpose_within_16x16_face)));
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

template <bool direct, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
ALWI void copy_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t dst_tile_index) {
    copy_check<direct, is_fp32_dest_acc_en, Format, Shape>();
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            direct>(detail::tile_address(src, itile))));
    MATH((llk_math_eltwise_unary_datacopy<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          is_fp32_dest_acc_en,
          BroadcastType::NONE,
          direct>(dst_tile_index)));
}

}  // namespace detail

// clang-format off
/**
 * Id-free datacopy init, Src path: unpack to SrcA, then an FPU A2D datacopy into DST. Takes an LLKOperand whose
 * data format + tile geometry are NTTPs (deduced from the argument); the register format is derived inside the LLK.
 * Reprograms the unpacker and math datacopy for the operand's format without touching the packer dst-offset
 * registers (those come from compute_kernel_hw_startup). Blackhole only.
 *
 * Any format whose register format is not 32-bit. A 32-bit register format (Float32 at fp32 Dest, Int32, UInt32)
 * does not fit the 19-bit SrcA and is rejected at compile time: use copy_direct_to_dest_init.
 *
 * fp8 (Fp8_e4m3 / Lf8 e5m2) is supported: the format-driven Src zero-substitution flag is programmed so fp8
 * zeros read back correctly, while other formats (e.g. bf16 -0.0 / 16b-int) keep their residual value.
 *
 * transpose / transpose_within_16x16_face support copy-with-transpose; both default off (0).
 * Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Template | is_fp32_dest_acc_en         | fp32 dest-accumulate mode                                    | bool        |                                                                     | False |
 * | Template | Format                      | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  | Register format not 32-bit                                          | True  |
 * | Template | Shape                       | Tile geometry (deduced from the LLKOperand argument)         | TensorShape |                                                                     | True  |
 * | Function | transpose                   | Flag to perform transpose on SrcA                           | uint32_t    | Any positive value will indicate transpose is set                   | False |
 * | Function | transpose_within_16x16_face | Flag to perform transpose within 16x16 face                 | uint32_t    | Any positive value will indicate transpose within 16x16 face is set | False |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_src_to_dest_init(
    LLKOperand<Format, Shape> /*src*/, std::uint32_t transpose = 0, std::uint32_t transpose_within_16x16_face = 0) {
    detail::copy_init<false, is_fp32_dest_acc_en, Format, Shape>(transpose, transpose_within_16x16_face);
}

// clang-format off
/**
 * Id-free datacopy init, direct path: the unpacker writes straight into DST (unpack-to-dest), all 32 bits. Same
 * operand handling as copy_src_to_dest_init. Blackhole only.
 *
 * Only a 32-bit register format: Float32 at fp32 Dest, Int32, UInt32 (compile-time checked). For any other format
 * use copy_src_to_dest_init.
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Template | is_fp32_dest_acc_en         | fp32 dest-accumulate mode                                    | bool        |                                                                     | False |
 * | Template | Format                      | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  | Float32 (fp32 Dest), Int32, UInt32                                  | True  |
 * | Template | Shape                       | Tile geometry (deduced from the LLKOperand argument)         | TensorShape |                                                                     | True  |
 * | Function | transpose                   | Flag to perform transpose of faces                          | uint32_t    | Any positive value will indicate transpose is set                   | False |
 * | Function | transpose_within_16x16_face | Flag to perform transpose within 16x16 face                 | uint32_t    | Any positive value will indicate transpose within 16x16 face is set | False |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_direct_to_dest_init(
    LLKOperand<Format, Shape> /*src*/, std::uint32_t transpose = 0, std::uint32_t transpose_within_16x16_face = 0) {
    detail::copy_init<true, is_fp32_dest_acc_en, Format, Shape>(transpose, transpose_within_16x16_face);
}

// clang-format off
/**
 * Id-free datacopy, Src path. Copies tile `itile` from the L1 buffer described by the LLKOperand through SrcA into
 * DST. `src.l1_address` is the buffer base; the unpack address is `tile_address(src, itile)`. Blackhole only.
 * Requires copy_src_to_dest_init, and the DST register to be acquired (tile_regs_acquire) before this call.
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        |         | False |
 * | Template | Format         | Buffer L1 data format (deduced from LLKOperand)        | DataFormat  | Register format not 32-bit | True |
 * | Template | Shape          | Tile geometry (deduced from LLKOperand)                | TensorShape |         | True |
 * | Function | src            | The source L1 operand (format+shape+buffer base)       | LLKOperand  |         | True |
 * | Function | itile          | Index of the tile within `src`, relative to its base   | uint32_t    | N/A     | True |
 * | Function | dst_tile_index | Tile index in the DST register                         | uint32_t    | Must be less than the acquired size of DST REG | True |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_src_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t dst_tile_index) {
    detail::copy_tile<false, is_fp32_dest_acc_en>(src, itile, dst_tile_index);
}

// clang-format off
/**
 * Id-free datacopy, direct path. Unpacks tile `itile` from the L1 buffer described by the LLKOperand straight into
 * DST. Blackhole only. Requires copy_direct_to_dest_init, and the DST register to be acquired (tile_regs_acquire).
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        |         | False |
 * | Template | Format         | Buffer L1 data format (deduced from LLKOperand)        | DataFormat  | Float32 (fp32 Dest), Int32, UInt32 | True |
 * | Template | Shape          | Tile geometry (deduced from LLKOperand)                | TensorShape |         | True |
 * | Function | src            | The source L1 operand (format+shape+buffer base)       | LLKOperand  |         | True |
 * | Function | itile          | Index of the tile within `src`, relative to its base   | uint32_t    | N/A     | True |
 * | Function | dst_tile_index | Tile index in the DST register                         | uint32_t    | Must be less than the acquired size of DST REG | True |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void copy_direct_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t dst_tile_index) {
    detail::copy_tile<true, is_fp32_dest_acc_en>(src, itile, dst_tile_index);
}

// clang-format off
/**
 * Id-free block datacopy, Src path, of a ct_dim x rt_dim block: source tile start_in_tile_index + i lands in DST
 * slot start_dst_tile_index + i. Loop form of copy_src_to_dest_tile (same block form as Quasar). Requires
 * copy_src_to_dest_init. No tails: for leftover tiles re-init, then run a second, shorter instantiation.
 *
 * | Param Type | Name                 | Description                                 | Type       | Valid Range | Required |
 * |------------|----------------------|---------------------------------------------|------------|-------------|----------|
 * | Template   | ct_dim / rt_dim      | Block columns / rows (tiles)                | uint32_t   | ct_dim * rt_dim <= acquired DST size | True |
 * | Template   | is_fp32_dest_acc_en  | fp32 dest-accumulate mode                   | bool       |             | False    |
 * | Function   | src                  | Source operand                              | LLKOperand | Register format not 32-bit | True |
 * | Function   | start_in_tile_index  | First source tile within src                | uint32_t   | N/A         | True     |
 * | Function   | start_dst_tile_index | First DST tile                              | uint32_t   | N/A         | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
ALWI void copy_src_to_dest_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_in_tile_index, std::uint32_t start_dst_tile_index = 0) {
    for (std::uint32_t i = 0; i < ct_dim * rt_dim; ++i) {
        detail::copy_tile<false, is_fp32_dest_acc_en>(src, start_in_tile_index + i, start_dst_tile_index + i);
    }
}

// clang-format off
/**
 * Id-free block datacopy, direct path, of a ct_dim x rt_dim block: source tile start_in_tile_index + i is unpacked
 * straight into DST slot start_dst_tile_index + i. Loop form of copy_direct_to_dest_tile. Requires
 * copy_direct_to_dest_init. No tails: for leftover tiles re-init, then run a second, shorter instantiation.
 *
 * | Param Type | Name                 | Description                                 | Type       | Valid Range | Required |
 * |------------|----------------------|---------------------------------------------|------------|-------------|----------|
 * | Template   | ct_dim / rt_dim      | Block columns / rows (tiles)                | uint32_t   | ct_dim * rt_dim <= acquired DST size | True |
 * | Template   | is_fp32_dest_acc_en  | fp32 dest-accumulate mode                   | bool       |             | False    |
 * | Function   | src                  | Source operand                              | LLKOperand | Float32 (fp32 Dest), Int32, UInt32 | True |
 * | Function   | start_in_tile_index  | First source tile within src                | uint32_t   | N/A         | True     |
 * | Function   | start_dst_tile_index | First DST tile                              | uint32_t   | N/A         | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
ALWI void copy_direct_to_dest_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_in_tile_index, std::uint32_t start_dst_tile_index = 0) {
    for (std::uint32_t i = 0; i < ct_dim * rt_dim; ++i) {
        detail::copy_tile<true, is_fp32_dest_acc_en>(src, start_in_tile_index + i, start_dst_tile_index + i);
    }
}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
