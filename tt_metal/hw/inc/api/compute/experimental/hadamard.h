// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute-API surface for the H128 (1x128) Hadamard transform.
//
// Computes Y = (1/sqrt(128)) * H_128 * x on a single tile (normalize=true,
// default) or Y = H_128 * x (normalize=false) using the FPU/MM path.
// See llk_math_hadamard.h for the algorithmic derivation.
//
// All operands are 1-face [16, 16] tiles; the inputs are bfloat16 and
// the result is packed out as bfp8 (bfloat8_b). The math is a custom
// narrow MOP: one (LoFi) or two (high-fidelity) MVMULs per matmul pass
// with a MOVD2B bridge, then an SFPU element-wise scale when normalize
// is true. H_16 is streamed into srcA bank 1 by the unpack thread
// (overlapping MM1), and the two passes write disjoint dst faces, so
// there is no MOVB2A copy and no ZEROACC.
//
// Tile-format assumptions (caller's responsibility):
//
//   h16_cb : single [16, 16] bfloat16 tile. H_16 fills the entire face.
//            Routed to srcB face (0, 0).
//
//   in_cb  : single [16, 16] bfloat16 tile. The 128 real input values
//            live in rows [0..8), cols [0..16). The unpack path zeroes
//            rows [8..16) itself (ZEROSRC over the full srcA span, see
//            llk_unpack_hadamard.h), so callers need not pad them.
//            Routed to srcA face (0, 0).
//
//   out_cb : single [16, 16] bfp8 (bfloat8_b) tile. The 1x128 result
//            lives in face 0 rows [0..8), cols [0..16). The caller
//            acquires the dst tile clean (zeroed) and runs one transform
//            per acquire; MM2 writes face 0, MM1's scratch lands in face 1.
//
// Fidelity: MATH_FIDELITY drives the math LLK. LoFi runs one MVMUL per
// pass; any high fidelity runs two per pass with an asymmetric phase
// step (MM1 +1, MM2 +2) that skips the zero-valued phases against the
// ±1 H_16 operand — 4 MVMULs total for full bf16 precision. See
// llk_math_hadamard.h for the phase derivation.

#pragma once

#include "api/compute/common.h"
#include "api/compute/reconfig_data_format.h"

// Blackhole-only: the H128 math/unpack LLKs live only in the Blackhole llk_lib.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_hadamard_api.h"
#endif

#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_hadamard_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Full initialization for the H128 Hadamard transform. Call once at the
 * top of a kernel before any hadamard_h128_tile call.
 *
 * The `normalize` template flag controls whether a post-MM2 SFPU pass
 * multiplies dst rows 0..7 by 1/sqrt(128). Default true; set false for
 * callers that apply their own normalization externally. Not supported
 * for fp32 dest.
 *
 * | Argument         | Description                                                            | Type     | Valid Range            | Required              |
 * |------------------|------------------------------------------------------------------------|----------|------------------------|-----------------------|
 * | in_cb_id         | The 1x128 input vector tile ([16, 16]; padding auto-zeroed on unpack)  | uint32_t | 0 to 31                | True                  |
 * | h16_cb_id        | The H_16 weight tile ([16, 16], H_16 fills the single face)            | uint32_t | 0 to 31                | True                  |
 * | out_cb_id        | The 1x128 output vector tile ([16, 16], result in rows 0..7)           | uint32_t | 0 to 31                | True                  |
 * | fp32_dest_acc_en | Whether to enable fp32 accumulation in dest                            | bool     | true/false             | False (default off)   |
 */
// clang-format on
template <bool normalize = true, bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hadamard_h128_init(const uint32_t in_cb_id, const uint32_t h16_cb_id, const uint32_t out_cb_id) {
    static_assert(!(normalize && fp32_dest_acc_en), "hadamard_h128 normalize=true is not supported with fp32 dest");

    // Configure both unpackers from h16's [16,16] single-face geometry so the
    // unpacker config is INDEPENDENT of the input CB's tile shape: srcA reads
    // the input data with h16's [16,16] config (a 4x[1,32] feed is read as 8
    // rows of 16), srcB reads h16. Both operands are bf16. in_cb_id is passed
    // to the init only for its single-face assertion.
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(h16_cb_id, h16_cb_id)));
    UNPACK((llk_unpack_hadamard_h128_init(h16_cb_id, in_cb_id, /*h16_tile_index=*/0)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(h16_cb_id, h16_cb_id)));
    MATH((llk_math_hadamard_h128_init<MATH_FIDELITY, normalize>()));

    // Fully re-establish the packer HW config + dest layout for out_cb_id.
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_dest_init<fp32_dest_acc_en, ckernel::PackMode::Default>()));
}

// clang-format off
/**
 * Perform one H128 Hadamard transform on the current tile. When
 * normalize=true (default), the result is scaled by 1/sqrt(128) in-place
 * on the SFPU after the two matmul passes.
 *
 * | Argument         | Description                                                  | Type     | Valid Range                                    | Required              |
 * |------------------|--------------------------------------------------------------|----------|------------------------------------------------|-----------------------|
 * | in_cb_id         | CB holding the 1x128 input tile (padding auto-zeroed)        | uint32_t | 0 to 31                                        | True                  |
 * | h16_cb_id        | CB holding the H_16 weight tile (H_16 fills the face)        | uint32_t | 0 to 31                                        | True                  |
 * | in_tile_index    | Index of the input tile within in_cb_id                      | uint32_t | < CB size                                      | True                  |
 * | h16_tile_index   | Index of the H_16 tile within h16_cb_id                      | uint32_t | < CB size                                      | True                  |
 * | dst_index        | DST register index that will receive the result Y            | uint32_t | < acquired DST size                            | True                  |
 */
// clang-format on
template <bool normalize = true>
ALWI void hadamard_h128_tile(
    const uint32_t in_cb_id,
    const uint32_t h16_cb_id,
    const uint32_t in_tile_index,
    const uint32_t h16_tile_index,
    const uint32_t dst_index) {
    // Single-face Hadamard unpack: phase 1 (context 0) h16 -> srcB,
    // input -> srcA bank 0 (zeroed + narrowed to 8 rows); phase 2 streams
    // H_16 into srcA bank 1 (overlaps MM1) so MM2 needs no MOVB2A. See
    // llk_unpack_hadamard.h.
    UNPACK((llk_unpack_hadamard_h128(h16_cb_id, in_cb_id, h16_tile_index, in_tile_index)));

    MATH((llk_math_hadamard_h128<MATH_FIDELITY, normalize>(dst_index)));
}

inline void hadamard_h128_uninit() { MATH((llk_math_hadamard_h128_uninit())); }

#endif

}  // namespace ckernel
