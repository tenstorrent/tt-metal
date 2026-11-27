// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
// SFPU scalar broadcast from DEST (the Compute API's sfpu_bcast_scalar_* / sfpu_mul_bcast_scalar)
// ============================================================================
//
// Element [0, 0] of a DEST tile is spread to every SFPU lane and parked in programmable constant LREG14 (sfpi's
// vConstFloatPrgm2); DEST tiles are then multiplied by it in place, as many as the caller likes. The parked value
// lives until something programs LREG12..14 (exp_tile_init, log_init, sfpu_normalize_bcast_scalar and the like).
//
// Quasar has no lane-id register, so the load keeps SFPU column 0 with the column broadcast's bit mask (the other
// columns become bit zero) and spreads it over the 8 columns by rotate-and-OR, as the column broadcast does: -0.0
// and NaN payloads survive. SFPCONFIG then writes LREG14 from row 0 of LREG0, per column, into all four rows.
//
// The multiply is the column broadcast's instruction, SFPMUL by the broadcast register with a zero addend, in
// groups of up to four DEST vectors (SFPLOADs into LREG0..3, SFPMULs, SFPSTOREs). DEST addresses are in the units
// of ckernel_sfpu_binary_bcast.h: 64 per 32x32 tile, 16 per face, 2 per sfpi vector. fp32 DEST only.

#include <cstdint>
#include <utility>

#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_binary_bcast.h"
#include "llk_defs.h"

namespace ckernel {
namespace sfpu {

constexpr std::uint32_t BCAST_SCALAR_VECTOR_STRIDE = 2;  // DEST address units per sfpi vector
constexpr std::uint32_t BCAST_SCALAR_GROUP = 4;          // vectors per SFPLOAD / SFPMUL / SFPSTORE group

/**
 * @brief Init for the parked-scalar multiply: nothing beyond the common SFPU init.
 *
 * The loads and stores use ADDR_MOD_7, which _llk_math_eltwise_sfpu_init_ programs with a zero increment.
 */
inline void _sfpu_bcast_scalar_init_() {}

/**
 * @brief Park element [0, 0] of DEST tile `dst_index_scalar` in every lane of LREG14.
 *
 * The binary-params signature (the second and third indices are unused) so the Compute API runs it through
 * SFPU_BINARY_CALL under VectorMode::None.
 *
 * @note Clobbers LREG0, LREG2 and LREG6 (Quasar's column-broadcast calls rebuild their lane mask on every call).
 *       Pair with @ref _calculate_mul_bcast_scalar_.
 */
inline void _sfpu_bcast_scalar_load_(
    const std::uint32_t dst_index_scalar,
    [[maybe_unused]] const std::uint32_t unused_in1,
    [[maybe_unused]] const std::uint32_t unused_out) {
    binary_bcast_scalar_to_lreg14_(dst_index_scalar * BCAST_DEST_TILE_STRIDE);
}

// N consecutive DEST vectors from `addr` times LREG14, in place, in LREG0..N-1. The SFPMUL -> SFPSTORE distance
// is kept at four instructions (NOPs for N < 4), the spacing of the column broadcast's band body.
template <std::uint32_t... I>
inline void _mul_bcast_scalar_group_(const std::uint32_t addr, std::integer_sequence<std::uint32_t, I...>) {
    constexpr std::uint32_t N = sizeof...(I);
    static_assert(N >= 1 && N <= BCAST_SCALAR_GROUP, "a group is 1..4 DEST vectors");
    (TT_SFPLOAD(
         p_sfpu::LREG0 + I, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, addr + I * BCAST_SCALAR_VECTOR_STRIDE),
     ...);
    (TTI_SFPMUL(p_sfpu::LREG0 + I, BCAST_SCALAR_LREG_PARKED, p_sfpu::LCONST_0, p_sfpu::LREG0 + I, 0 /*instr_mod1*/),
     ...);
    for (std::uint32_t i = N; i < BCAST_SCALAR_GROUP; i++) {
        TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    }
    (TT_SFPSTORE(
         p_sfpu::LREG0 + I, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, addr + I * BCAST_SCALAR_VECTOR_STRIDE),
     ...);
}

template <std::uint32_t N>
inline void _mul_bcast_scalar_group_(const std::uint32_t addr) {
    _mul_bcast_scalar_group_(addr, std::make_integer_sequence<std::uint32_t, N>{});
}

// The first ITERATIONS vectors of the face at `face_addr`: iteration k is face rows 4 * (k / 2) .. +3, one column
// parity (the convention of every SFPU body), so 2 iterations are rows 0..3 of the face and 8 the whole face.
template <int ITERATIONS>
inline void _mul_bcast_scalar_face_(const std::uint32_t face_addr) {
    static_assert(ITERATIONS >= 1 && ITERATIONS <= 8, "1..8 vectors per face");
    if constexpr (ITERATIONS > static_cast<int>(BCAST_SCALAR_GROUP)) {
        _mul_bcast_scalar_group_<BCAST_SCALAR_GROUP>(face_addr);
        _mul_bcast_scalar_group_<ITERATIONS - BCAST_SCALAR_GROUP>(
            face_addr + BCAST_SCALAR_GROUP * BCAST_SCALAR_VECTOR_STRIDE);
    } else {
        _mul_bcast_scalar_group_<ITERATIONS>(face_addr);
    }
}

/**
 * @brief DEST[dst_index] *= LREG14 over the faces `vector_mode` names and the first ITERATIONS vectors of each.
 *
 * RC: all four faces; R: faces 0 and 1 (tile rows 0..15); C: faces 0 and 2 (tile columns 0..15). Runs once under
 * VectorMode::None; the binary-params signature (second and third indices unused).
 *
 * @tparam ITERATIONS sfpi vectors per face, 1..8.
 * @note Call @ref _sfpu_bcast_scalar_load_ first. Clobbers LREG0..3.
 */
template <int ITERATIONS>
inline void _calculate_mul_bcast_scalar_(
    const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t unused_in1,
    [[maybe_unused]] const std::uint32_t unused_out,
    const VectorMode vector_mode) {
    const std::uint32_t base = dst_index * BCAST_DEST_TILE_STRIDE;
    const bool upper_right = vector_mode == VectorMode::RC || vector_mode == VectorMode::R;
    const bool lower_left = vector_mode == VectorMode::RC || vector_mode == VectorMode::C;
    const bool lower_right = vector_mode == VectorMode::RC;
    _mul_bcast_scalar_face_<ITERATIONS>(base);
    if (upper_right) {
        _mul_bcast_scalar_face_<ITERATIONS>(base + BCAST_FACE_STRIDE);
    }
    if (lower_left) {
        _mul_bcast_scalar_face_<ITERATIONS>(base + BCAST_FACE_PAIR_STRIDE);
    }
    if (lower_right) {
        _mul_bcast_scalar_face_<ITERATIONS>(base + BCAST_FACE_PAIR_STRIDE + BCAST_FACE_STRIDE);
    }
}

}  // namespace sfpu
}  // namespace ckernel
