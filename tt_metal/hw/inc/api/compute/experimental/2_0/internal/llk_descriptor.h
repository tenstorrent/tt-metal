// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"                // ckernel::TensorShape + geometry helpers
#include "api/compute/common_globals.h"  // DataFormat enum + ckernel_defs macros (IS_BFP_FORMAT, SCALE_DATUM_SIZE, ...)

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
 * t*stride). The id-free stand-in for the CB's fifo_page_size, which the shipping factories set to one tile's
 * size. Geometry-exact for BOTH branches, so partial / tiny tiles stride correctly:
 *   * Block floats (Bfp8/Bfp4/Bfp2): a tile is [packed mantissas | shared-exponent section]. This mirrors
 *     tt_metal Tile::get_tile_size (impl/data_format/tile.cpp) scaled by the real geometry: mantissa =
 *     total_tensor_size() at the format's storage width (Bfp8 1 B/datum, Bfp4 1/2, Bfp2 1/4), exp section =
 *     round_up(face_r_dim * total_num_faces, 16). GET_L1_HEADERLESS_TILE_SIZE hard-codes the full-32x32 value
 *     and is WRONG for a partial BFP tile (fewer face rows / faces), so we compute the size here instead.
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

// NOTE (matmul partial_face): the id-free matmul derives partial_face inline at its two call sites, NOT via a
// shared helper, because the UNPACK and MATH engines use DIFFERENT thresholds (inherited from the legacy CB-id
// path, do not "unify"):
//   * MATH  (llk_math_matmul.h):        partial_face = (shape.total_row_dim() < FACE_R_DIM)  -- == legacy math init.
//   * UNPACK (llk_unpack_AB_matmul.h):  partial_face = (shape.total_row_dim() < TILE_R_DIM)  -- == the legacy
//     host-side Tile::partial_face (tile height < TILE_HEIGHT) fed via get_operand_partial_face().
// A one-face-high operand (total_row_dim() == 16) is thus partial-face to the unpacker (16 < 32) but full-face to
// the math (16 == FACE_R_DIM, not < ).

/**
 * (internal) Per-tile L1 size in 16B words (fifo_page_size units) for a matmul operand descriptor: geometry-exact
 * for linear formats, exp section included for block floats. Thin wrapper over tile_stride_words. `desc`: the
 * operand's descriptor.
 */
constexpr std::uint32_t matmul_tile_size(const LLKMemDescriptor& desc) {
    return tile_stride_words(desc.format, desc.shape);
}

// -----------------------------------------------------------------------------------------------------
// Compile-time contract helpers (PART D). With no CB id, the legality the CB descriptor used to guarantee is
// re-established as static_asserts on the operand geometry NTTPs, firing at the user's call site.
// -----------------------------------------------------------------------------------------------------

/**
 * (internal) True iff the tile shape is one the HW can address: face_r_dim in {1,2,4,8,16}, face_c_dim == 16
 * (always 16 in hardware), the face grid within HW limits (num_faces_r_dim / num_faces_c_dim each in {1,2},
 * i.e. MAX_NUM_FACES_R_DIM / MAX_NUM_FACES_C_DIM), and total faces in {1,2,4}. Superset of the canonical
 * runtime validator ckernel::validate_tensor_shape_tile_dependent_ops_ (tensor_shape.h): it adds the per-axis
 * face-grid bound the runtime validator omits, so shapes like {16,16,4,1} / {16,16,1,4} (a 64x16 / 16x64 tile
 * that exceeds the 32x32 limit yet has total_num_faces()==4) are rejected here rather than silently addressed.
 * constexpr so it is usable in the compute-op static_asserts (PART D). `s`: the tile geometry to check.
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
 * (internal) True iff two operands carry the same tile geometry (a two-input op requires matching shapes).
 * Used by the binary-op static_asserts (PART D). `a`, `b`: the two tile geometries to compare.
 */
constexpr bool same_tile_shape(TensorShape a, TensorShape b) {
    return a.face_r_dim == b.face_r_dim && a.face_c_dim == b.face_c_dim && a.num_faces_r_dim == b.num_faces_r_dim &&
           a.num_faces_c_dim == b.num_faces_c_dim;
}

}  // namespace experimental
}  // namespace ckernel
