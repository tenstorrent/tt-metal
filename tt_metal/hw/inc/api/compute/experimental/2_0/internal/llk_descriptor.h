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

// The compile-time descriptor the LLK APIs accept as an NTTP: buffer L1 format + tile geometry.
struct LLKMemDescriptor {
    std::uint8_t format;  // buffer L1 format (what the unpacker reads / the packer writes)
    TensorShape shape;    // tile geometry; derive num_faces / tile dims via TensorShape helpers
};

// Per-tile L1 stride in 16-byte words, for absolute (out-of-order) tile addressing (base + t*stride). This is
// the id-free stand-in for the CB's fifo_page_size, which the shipping factories set to one tile's size.
//   * Block floats (Bfp8/Bfp4/Bfp2): a tile carries a shared-exponent section that SCALE_DATUM_SIZE omits
//     (and it miscounts the sub-byte mantissa), so use GET_L1_HEADERLESS_TILE_SIZE -- exp section included,
//     matching tile_size(fmt) / llk_pack_fast_tilize. It assumes a full 32x32 tile (partial BFP tiles are
//     out of scope; no such path today).
//   * Linear formats (Float32/Float16/int): keep the geometry-exact datum-count size so tiny tiles
//     (face_r_dim / num_faces below full) stride correctly.
constexpr std::uint32_t tile_stride_words(std::uint8_t format, TensorShape shape) {
    return IS_BFP_FORMAT(format) ? GET_L1_HEADERLESS_TILE_SIZE(format)
                                 : (SCALE_DATUM_SIZE(format, shape.total_tensor_size()) >> 4);
}

// -----------------------------------------------------------------------------------------------------
// Compile-time contract helpers (PART D). With no CB id, the legality the CB descriptor used to guarantee is
// re-established as static_asserts on the operand geometry NTTPs, firing at the user's call site.
// -----------------------------------------------------------------------------------------------------

// A tile shape the HW can address: face_r_dim in {1,2,4,8,16}, total faces in {1,2,4}.
constexpr bool is_legal_tile_shape(TensorShape s) {
    const bool fr =
        (s.face_r_dim == 1 || s.face_r_dim == 2 || s.face_r_dim == 4 || s.face_r_dim == 8 || s.face_r_dim == 16);
    const std::uint8_t nf = s.total_num_faces();
    return fr && (nf == 1 || nf == 2 || nf == 4);
}

// Whether two operands carry the same tile geometry (a two-input op requires matching shapes).
constexpr bool same_tile_shape(TensorShape a, TensorShape b) {
    return a.face_r_dim == b.face_r_dim && a.face_c_dim == b.face_c_dim && a.num_faces_r_dim == b.num_faces_r_dim &&
           a.num_faces_c_dim == b.num_faces_c_dim;
}

}  // namespace experimental
}  // namespace ckernel
