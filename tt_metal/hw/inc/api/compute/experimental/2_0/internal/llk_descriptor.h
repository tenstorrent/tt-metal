// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"                // ckernel::TensorShape + geometry helpers
#include "api/compute/common_globals.h"  // DataFormat enum + ckernel_defs macros (IS_BFP_FORMAT, SCALE_DATUM_SIZE, ...)
#include "data_format_derive.h"          // infer_unpack_dst_format (register-format rebias)

// =====================================================================================================
// INTERNAL (NOT user-facing). The compile-time descriptor + per-tile stride math consumed by the LLK layer
// AND the compute-op layer. Kernel authors never touch this header -- they use LLKOperand (llk_operand.h).
//
//   * LLKMemDescriptor -- the compile-time "sticky note" the LLK ops consume as an NTTP (buffer L1 data
//     format + tile geometry). Passed as a non-type template parameter (-ftt-nttp) so the per-format
//     switches / register writes / asserts fold and DCE away.
//   * tile_stride_words -- the per-tile L1 stride (in 16-byte words) used for absolute tile addressing.
//
// The register-side format is NOT carried (derived inside the LLK from `format`); there is NO CB id and NO
// knowledge of the source (CB / DataflowBuffer / Scratchpad / LocalTensorAccessor).
// =====================================================================================================

namespace ckernel {
namespace experimental {

/**
 * (internal) The compile-time descriptor the LLK APIs accept as an NTTP: buffer L1 format + tile geometry.
 * `format` is what the unpacker reads / the packer writes; `shape` is the tile geometry (derive num_faces /
 * tile dims via the TensorShape helpers).
 */
struct LLKMemDescriptor {
    DataFormat format;  // buffer L1 format (what the unpacker reads / the packer writes)
    TensorShape shape;  // tile geometry; derive num_faces / tile dims via TensorShape helpers
};

// Round up to the 16-byte L1 word granularity (matches tt_metal's L1_ALIGNMENT on Blackhole).
constexpr std::uint32_t round_up_l1_words(std::uint32_t bytes) { return (bytes + 15u) >> 4; }

// clang-format off
/**
 * (internal) Per-tile L1 stride in 16-byte words, for absolute (out-of-order) tile addressing (base +
 * t*stride). The id-free stand-in for the CB's fifo_page_size, which the shipping factories set to one
 * tile's size. Geometry-exact for both branches, so partial / tiny tiles stride correctly:
 *   * Block floats (Bfp8/Bfp4/Bfp2): a tile is [packed mantissas | shared-exponent section], scaled by
 *     geometry to match tt_metal Tile::get_tile_size -- mantissa bytes scale with storage width (Bfp8
 *     1 B/datum, Bfp4 1/2, Bfp2 1/4), exponent section is round_up(face_r_dim * total_num_faces, 16).
 *     (GET_L1_HEADERLESS_TILE_SIZE assumes a full 32x32 tile and is unsafe for a partial BFP tile.)
 *   * Linear formats (Float32/Float16/int): the geometry-exact datum-count size (SCALE_DATUM_SIZE >> 4).
 *
 * | Param Type | Name   | Description                       | Type        | Valid Range | Required |
 * |------------|--------|-----------------------------------|-------------|-------------|----------|
 * | Function   | format | Buffer L1 data format             | DataFormat  | N/A         | True     |
 * | Function   | shape  | Tile geometry                     | TensorShape | N/A         | True     |
 */
// clang-format on
constexpr std::uint32_t tile_stride_words(DataFormat format, TensorShape shape) {
    // The size macros are the legacy numeric-format path; convert the typed format once here.
    const std::uint32_t fmt = static_cast<std::uint32_t>(format);
    if (!IS_BFP_FORMAT(fmt)) {
        return SCALE_DATUM_SIZE(fmt, shape.total_tensor_size()) >> 4;
    }
    // Block-float tile size, generalized from Tile::get_tile_size to the tile geometry (partial BFP support).
    const std::uint32_t datums = shape.total_tensor_size();
    // Shared-exponent section: one exponent byte per face row across all faces, padded to the L1 word.
    // (== tt_metal round_up(face_shape[0] * num_faces, L1_ALIGNMENT).)
    const std::uint32_t exp_bytes =
        (static_cast<std::uint32_t>(shape.face_r_dim) * shape.total_num_faces() + 15u) & ~15u;
    std::uint32_t mantissa_bytes = datums;  // Bfp8/Bfp8_b: 1 byte per datum
    switch (masked_data_format(fmt)) {
        case to_underlying(DataFormat::Bfp4):
        case to_underlying(DataFormat::Bfp4_b): mantissa_bytes = datums >> 1; break;  // 4-bit mantissas
        case to_underlying(DataFormat::Bfp2):
        case to_underlying(DataFormat::Bfp2_b): mantissa_bytes = datums >> 2; break;  // 2-bit mantissas
        default: break;                                                               // Bfp8 / Bfp8_b
    }
    return round_up_l1_words(mantissa_bytes + exp_bytes);
}

// NOTE (matmul partial_face): the id-free matmul computes partial_face inline at its two call sites, NOT via
// a shared helper, because UNPACK and MATH use DIFFERENT thresholds:
//   * MATH  (llk_math_matmul.h):       partial_face = (shape.total_row_dim() < FACE_R_DIM).
//   * UNPACK (llk_unpack_AB_matmul.h): partial_face = (shape.total_row_dim() < TILE_R_DIM), matching the
//     host-side Tile::partial_face (tile height < TILE_HEIGHT) via get_operand_partial_face().
// A one-face-high operand (total_row_dim() == 16) is thus partial-face to the unpacker (16 < 32) but
// full-face to the math (16 == FACE_R_DIM, not <). Matmul per-tile L1 size is tile_stride_words.

// -----------------------------------------------------------------------------------------------------
// Compile-time contract helpers. With no CB id, the legality the CB descriptor used to guarantee is
// re-established as static_asserts on the operand geometry NTTPs, firing at the user's call site.
// -----------------------------------------------------------------------------------------------------

/**
 * (internal) True iff the tile shape is one the HW can address: face_r_dim in {1,2,4,8,16}, face_c_dim == 16
 * (always 16 in hardware), the face grid within HW limits (num_faces_r_dim / num_faces_c_dim each in {1,2},
 * i.e. MAX_NUM_FACES_R_DIM / MAX_NUM_FACES_C_DIM), and total faces in {1,2,4}. Superset of the canonical
 * runtime validator ckernel::validate_tensor_shape_tile_dependent_ops_ (tensor_shape.h): it adds the per-axis
 * face-grid bound the runtime validator omits, so shapes like {16,16,4,1} / {16,16,1,4} (a 64x16 / 16x64 tile
 * that exceeds the 32x32 limit yet has total_num_faces()==4) are rejected here rather than silently addressed.
 * constexpr so it is usable in the compute-op static_asserts. `s`: the tile geometry to check.
 */
constexpr bool is_legal_tile_shape(TensorShape s) {
    const bool fr =
        (s.face_r_dim == 1 || s.face_r_dim == 2 || s.face_r_dim == 4 || s.face_r_dim == 8 || s.face_r_dim == 16);
    const bool grid =
        (s.num_faces_r_dim == 1 || s.num_faces_r_dim == 2) && (s.num_faces_c_dim == 1 || s.num_faces_c_dim == 2);
    const std::uint8_t nf = s.total_num_faces();
    return fr && (s.face_c_dim == 16) && grid && (nf == 1 || nf == 2 || nf == 4);
}

/**
 * (internal) True iff `f` is a 32-bit register format (Float32/UInt32/Int32). These must take the
 * unpack-to-dest A2D path (SrcB is only 19 bits wide), so bcast / transpose use this to select that path at
 * compile time (folds to a constant). Shared so the same 3-way format compare is not spelled out per site.
 */
constexpr bool is_32bit_format(DataFormat f) {
    return f == DataFormat::Float32 || f == DataFormat::UInt32 || f == DataFormat::Int32;
}

/**
 * (internal) True iff the operand's REGISTER format (after fp32-dest-acc rebias) is 32-bit and must take
 * the unpack-to-dest A2D path. Shared by unary bcast and transpose so the same infer+is_32bit check is
 * not spelled twice.
 */
template <DataFormat Format, bool is_fp32_dest_acc_en>
constexpr bool is_unpack_to_dest() {
    return is_32bit_format(ckernel::infer_unpack_dst_format(Format, is_fp32_dest_acc_en));
}

/**
 * (internal) True iff `f` is a block-float format (Bfp8/Bfp4/Bfp2, a- and b-exponent families).
 * The BH compute datapath only handles FULL-height (32-row) block-float tiles; partial-height block floats
 * are rejected at the copy/pack surfaces (silent-garbage otherwise). Typed wrapper over IS_BFP_FORMAT.
 */
constexpr bool is_block_float_format(DataFormat f) { return IS_BFP_FORMAT(static_cast<std::uint32_t>(f)); }

/**
 * (internal) True iff `f` is an 8-bit L1 format (Int8/UInt8/Fp8_e4m3/Lf8). The unpack tilize datapath
 * handles these through a dedicated face-addressing sequence that only supports a two-face-wide tile row (it
 * addresses face 1 as the right-hand neighbour), so a narrow (single face column) 8-bit tile is rejected at
 * the tilize surface. Typed wrapper over IS_8BIT_FORMAT.
 */
constexpr bool is_8bit_format(DataFormat f) { return IS_8BIT_FORMAT(static_cast<std::uint32_t>(f)); }

/**
 * (internal) True iff the tile has fewer than 32 rows (partial tile height). For block-float formats a
 * partial-height tile does not flow correctly through the BH compute datapath; use with is_block_float_format
 * to reject that combination at compile time.
 */
constexpr bool is_partial_height(TensorShape s) { return s.total_row_dim() < TILE_R_DIM; }

/**
 * (internal) True iff the Blackhole tilize datapath has unpack and pack programs for an input tile of format
 * `in_format` and geometry `in` tilized into an output tile of geometry `out`:
 *   - the input must have 2 or 4 faces (there is no single-face tilize program);
 *   - an 8-bit (Int8/UInt8/Fp8_e4m3/Lf8) input must be two faces wide (its face-addressing sequence reads face 1
 *     as the right-hand neighbour, so a narrow single-face-column 8-bit tile reads the wrong face);
 *   - for a non-8-bit input the packer runs its Tilize program from the OUTPUT geometry, and that program's
 *     single-row-face form (one PACR packing both faces of a 1x32 row) exists for exactly two faces: with four
 *     one-row faces it is run once per face pair and the second pair is packed from the wrong DEST rows. An 8-bit
 *     input switches the packer to its Default program, which has no such limit.
 * All three limits are runtime LLK_ASSERTs in the LLK that production builds compile out; the tilize ops assert
 * this at compile time.
 */
constexpr bool is_supported_tilize_operands(DataFormat in_format, TensorShape in, TensorShape out) {
    const std::uint8_t nf = in.total_num_faces();
    const bool face_count_ok = (nf == 2 || nf == 4);
    const bool width_ok = !(is_8bit_format(in_format) && in.num_faces_c_dim == 1);
    const bool pack_one_row_faces_ok = is_8bit_format(in_format) || out.face_r_dim != 1 || out.total_num_faces() == 2;
    return face_count_ok && width_ok && pack_one_row_faces_ok;
}

/**
 * (internal) True iff the Blackhole matmul LLK has unpack + math programs for the operand pair (`in0` -> SrcB,
 * `in1` -> SrcA). The math programs are selected from the operands' TOTAL dimensions and assume the face grid
 * behind them, so both are checked:
 *   - the inner dimension must match (in0 columns == in1 rows);
 *   - in0 must be K == 32 wide (any legal 1..32-row height, tiny tiles included) or the 32x16 tile -- a 16-column
 *     in0 with fewer than 32 rows (16x16 and its sub-16-row forms) has no math program and would run the default
 *     32x32 one over faces it does not have;
 *   - an in0 with faces shorter than 16 rows must be a single face row: an in0 shorter than 32 rows is unpacked
 *     face by face, two faces per tile, and the math treats a 16-row in0 as two 16-row faces (a {8,16,2,2} in0
 *     would have half its faces unread);
 *   - in1 must be 32x32, 32x16 or 16x32 with full-height (16-row) faces: a 16x32 in1 is likewise unpacked as two
 *     faces, which the math program takes as two 16-row faces.
 * Matches the support matrix the LLK's own test validator enforces (tt-llk tests/python_tests/fuser/validator.py,
 * MATMUL_INNER_TILE_DIMS + SUPPORTED_TILE_SIZES, which only lists 16-row-face in1 tiles). The LLK's only guard is
 * a runtime LLK_ASSERT on the 16x16-by-16x16 pair that production builds compile out.
 */
constexpr bool is_supported_matmul_operand_pair(TensorShape in0, TensorShape in1) {
    const bool inner_dim_match = in0.total_col_dim() == in1.total_row_dim();
    const bool in0_ok = (in0.total_col_dim() == TILE_C_DIM) || (in0.total_row_dim() == TILE_R_DIM);
    const bool in0_face_grid_ok = in0.face_r_dim == FACE_R_DIM || in0.num_faces_r_dim == 1;
    const bool in1_face_grid_ok = in1.face_r_dim == FACE_R_DIM;
    const bool in1_full_height =
        in1.total_row_dim() == TILE_R_DIM && (in1.total_col_dim() == TILE_C_DIM || in1.total_col_dim() == FACE_C_DIM);
    const bool in1_half_height = in1.total_row_dim() == FACE_R_DIM && in1.total_col_dim() == TILE_C_DIM;
    return inner_dim_match && in0_ok && in0_face_grid_ok && in1_face_grid_ok && (in1_full_height || in1_half_height);
}

/**
 * (internal) True iff two operands carry the same tile geometry (a two-input op requires matching shapes).
 * Used by the binary-op static_asserts. `a`, `b`: the two tile geometries to compare.
 */
constexpr bool same_tile_shape(TensorShape a, TensorShape b) {
    return a.face_r_dim == b.face_r_dim && a.face_c_dim == b.face_c_dim && a.num_faces_r_dim == b.num_faces_r_dim &&
           a.num_faces_c_dim == b.num_faces_c_dim;
}

}  // namespace experimental
}  // namespace ckernel
