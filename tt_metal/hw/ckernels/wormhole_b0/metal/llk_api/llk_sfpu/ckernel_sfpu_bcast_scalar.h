// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
// SFPU scalar broadcast from DEST (the compute API's sfpu_bcast_scalar_* / sfpu_mul_bcast_scalar)
// ============================================================================
//
// Element [0, 0] of a DEST tile is spread to all 32 SFPU lanes and parked in programmable constant 2 (LREG14,
// sfpi's vConstFloatPrgm2); DEST tiles are then multiplied by it in place, as many as the caller likes, without a
// resident operand tile. The parked value is the same kind of hidden state as the column-broadcast lane mask: it
// lives until an init that programs the constant registers LREG12..14 (exp_tile_init, sfpu_normalize_bcast_scalar
// and the like), after which sfpu_bcast_scalar_init + sfpu_bcast_scalar_load bring it back.
//
// The load is bit-exact: lane 0 is kept by a lane-id predicate (the other lanes become +0.0), spread over its
// 8-lane sub-vector by rotate-by-1, 2, 4 and bitwise OR (the zero lanes contribute no bits, so -0.0, NaN payloads
// and denormals survive), then over the four sub-vectors by SFPTRANSP of four copies (output 0's sub-vector j is
// input j's sub-vector 0).
//
// The multiply is the column-broadcast kernel's own instruction, SFPMUL by the broadcast register with a zero
// addend, so a tile scaled by the parked scalar is bit-identical to sfpu_mul_bcast_col with a uniform column.
// Per group of up to four DEST vectors: SFPLOADs into LREG0..3, SFPMULs by LREG14, SFPSTOREs, the layout of
// _process_col_bcast_row_band_. DEST addresses are in the units of that kernel: 64 per 32x32 tile, 16 per face,
// 2 per sfpi vector. fp32 DEST only.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_sfpu_binary_bcast.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel {
namespace sfpu {

// The zero-increment address mode the straight SFPLOAD / SFPSTORE sequences of this file use (the one the
// column-broadcast kernel uses on this architecture).
constexpr std::uint32_t BCAST_SCALAR_ADDR_MOD = ADDR_MOD_3;

// sfpi vectors per 32x32 fp32 DEST tile (DEST_TILE_SIZE_RAW address units, two per vector).
constexpr std::uint32_t BCAST_SCALAR_VECS_PER_TILE = DEST_TILE_SIZE_RAW / 2;

inline void _sfpu_bcast_scalar_init_() {
    // The shared SFPU configuration (as the sibling inits) and the zero-stride address modifier the loads and
    // stores below address DEST through. Neither the column-broadcast mask nor the replay program is needed.
    _init_sfpu_config_reg();
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

sfpi_inline sfpi::vFloat _bcast_scalar_or_bits_(sfpi::vFloat a, sfpi::vFloat b) {
    return sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(a) | sfpi::as<sfpi::vUInt>(b));
}

// Element [0, 0] of DEST[dst_index_scalar] -> every lane of LREG14. The binary-params signature (the second and
// third indices are unused) so the compute API can run it through SFPU_BINARY_CALL under VectorMode::None.
inline void _sfpu_bcast_scalar_load_(
    std::uint32_t dst_index_scalar,
    [[maybe_unused]] std::uint32_t unused_in1,
    [[maybe_unused]] std::uint32_t unused_out) {
    sfpi::vFloat s = sfpi::dst_reg[dst_index_scalar * BCAST_SCALAR_VECS_PER_TILE];
    v_if(sfpi::vConstTileId != 0) { s = 0.0f; }  // tile id = 2 * lane: lane 0 only
    v_endif;
    sfpi::vFloat t = sfpi::subvec_shflror1(s);
    s = _bcast_scalar_or_bits_(s, t);
    t = sfpi::subvec_shflror1(s);
    t = sfpi::subvec_shflror1(t);
    s = _bcast_scalar_or_bits_(s, t);
    t = sfpi::subvec_shflror1(s);
    t = sfpi::subvec_shflror1(t);
    t = sfpi::subvec_shflror1(t);
    t = sfpi::subvec_shflror1(t);
    s = _bcast_scalar_or_bits_(s, t);
    sfpi::vFloat a = s, b = s, c = s, d = s, e = 0.0f, f = 0.0f, g = 0.0f, h = 0.0f;
    sfpi::subvec_transp(a, b, c, d, e, f, g, h);
    sfpi::vConstFloatPrgm2 = a;
}

// N consecutive DEST vectors from `addr` (address units) times LREG14, in place. No sfpi value is live here, so
// LREG0..3 are free. The SFPMUL -> SFPSTORE distance is kept at four instructions (NOPs for N < 4), the spacing
// the column-broadcast kernel relies on.
template <int N>
inline void _mul_bcast_scalar_group_(std::uint32_t addr) {
    static_assert(N >= 1 && N <= 4, "a group is 1..4 DEST vectors");
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TT_SFPLOAD(p_sfpu::LREG0, IM, BCAST_SCALAR_ADDR_MOD, addr);
    if constexpr (N > 1) {
        TT_SFPLOAD(p_sfpu::LREG1, IM, BCAST_SCALAR_ADDR_MOD, addr + 2);
    }
    if constexpr (N > 2) {
        TT_SFPLOAD(p_sfpu::LREG2, IM, BCAST_SCALAR_ADDR_MOD, addr + 4);
    }
    if constexpr (N > 3) {
        TT_SFPLOAD(p_sfpu::LREG3, IM, BCAST_SCALAR_ADDR_MOD, addr + 6);
    }
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    if constexpr (N > 1) {
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    }
    if constexpr (N > 2) {
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    }
    if constexpr (N > 3) {
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    }
    for (int i = N; i < 4; i++) {
        TTI_SFPNOP;
    }
    TT_SFPSTORE(p_sfpu::LREG0, IM, BCAST_SCALAR_ADDR_MOD, addr);
    if constexpr (N > 1) {
        TT_SFPSTORE(p_sfpu::LREG1, IM, BCAST_SCALAR_ADDR_MOD, addr + 2);
    }
    if constexpr (N > 2) {
        TT_SFPSTORE(p_sfpu::LREG2, IM, BCAST_SCALAR_ADDR_MOD, addr + 4);
    }
    if constexpr (N > 3) {
        TT_SFPSTORE(p_sfpu::LREG3, IM, BCAST_SCALAR_ADDR_MOD, addr + 6);
    }
}

// The first ITERATIONS vectors of the face at `face_addr`: iteration k is face rows 4 * (k / 2) .. +3, one column
// parity (the convention of every SFPU body), so 2 iterations are rows 0..3 of the face and 8 the whole face.
template <int ITERATIONS>
inline void _mul_bcast_scalar_face_(std::uint32_t face_addr) {
    static_assert(ITERATIONS >= 1 && ITERATIONS <= 8, "1..8 vectors per face");
    if constexpr (ITERATIONS > 4) {
        _mul_bcast_scalar_group_<4>(face_addr);
        _mul_bcast_scalar_group_<ITERATIONS - 4>(face_addr + 8);
    } else {
        _mul_bcast_scalar_group_<ITERATIONS>(face_addr);
    }
}

// The same group with a compile-time DEST address: every SFPLOAD / SFPSTORE then carries an immediate, and the SFPU
// issues about one instruction per cycle; with a run-time address each instruction word is assembled first and the
// sequence runs at about half that rate (measured on the GDN decode update pass, 16 tiles per token).
template <int N, std::uint32_t ADDR>
inline void _mul_bcast_scalar_group_imm_() {
    static_assert(N >= 1 && N <= 4, "a group is 1..4 DEST vectors");
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    TT_SFPLOAD(p_sfpu::LREG0, IM, BCAST_SCALAR_ADDR_MOD, ADDR);
    if constexpr (N > 1) {
        TT_SFPLOAD(p_sfpu::LREG1, IM, BCAST_SCALAR_ADDR_MOD, ADDR + 2);
    }
    if constexpr (N > 2) {
        TT_SFPLOAD(p_sfpu::LREG2, IM, BCAST_SCALAR_ADDR_MOD, ADDR + 4);
    }
    if constexpr (N > 3) {
        TT_SFPLOAD(p_sfpu::LREG3, IM, BCAST_SCALAR_ADDR_MOD, ADDR + 6);
    }
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    if constexpr (N > 1) {
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    }
    if constexpr (N > 2) {
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    }
    if constexpr (N > 3) {
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    }
    for (int i = N; i < 4; i++) {
        TTI_SFPNOP;
    }
    TT_SFPSTORE(p_sfpu::LREG0, IM, BCAST_SCALAR_ADDR_MOD, ADDR);
    if constexpr (N > 1) {
        TT_SFPSTORE(p_sfpu::LREG1, IM, BCAST_SCALAR_ADDR_MOD, ADDR + 2);
    }
    if constexpr (N > 2) {
        TT_SFPSTORE(p_sfpu::LREG2, IM, BCAST_SCALAR_ADDR_MOD, ADDR + 4);
    }
    if constexpr (N > 3) {
        TT_SFPSTORE(p_sfpu::LREG3, IM, BCAST_SCALAR_ADDR_MOD, ADDR + 6);
    }
}
template <int ITERATIONS, std::uint32_t FACE_ADDR>
inline void _mul_bcast_scalar_face_imm_() {
    if constexpr (ITERATIONS > 4) {
        _mul_bcast_scalar_group_imm_<4, FACE_ADDR>();
        _mul_bcast_scalar_group_imm_<ITERATIONS - 4, FACE_ADDR + 8>();
    } else {
        _mul_bcast_scalar_group_imm_<ITERATIONS, FACE_ADDR>();
    }
}
// One tile at a compile-time base address over the faces the vector mode names.
template <int ITERATIONS, std::uint32_t BASE>
inline void _mul_bcast_scalar_tile_imm_(VectorMode vector_mode) {
    const bool upper_right = vector_mode == VectorMode::RC || vector_mode == VectorMode::R;
    const bool lower_left = vector_mode == VectorMode::RC || vector_mode == VectorMode::C;
    const bool lower_right = vector_mode == VectorMode::RC;
    _mul_bcast_scalar_face_imm_<ITERATIONS, BASE + FACE0_BASE>();
    if (upper_right) {
        _mul_bcast_scalar_face_imm_<ITERATIONS, BASE + FACE1_BASE>();
    }
    if (lower_left) {
        _mul_bcast_scalar_face_imm_<ITERATIONS, BASE + FACE2_BASE>();
    }
    if (lower_right) {
        _mul_bcast_scalar_face_imm_<ITERATIONS, BASE + FACE3_BASE>();
    }
}
// DEST[dst_index] <- DEST[dst_index] * LREG14 over the faces the vector mode names (RC: all four; R: faces 0 and
// 1, tile rows 0..15; C: faces 0 and 2, tile columns 0..15) and the first ITERATIONS vectors of each. Runs once
// under VectorMode::None; the binary-params signature (second and third indices unused). Slots 0..7 (an fp32 DEST
// holds at most eight) take the compile-time address form; any other index the run-time one.
template <int ITERATIONS>
inline void _calculate_mul_bcast_scalar_(
    std::uint32_t dst_index,
    [[maybe_unused]] std::uint32_t unused_in1,
    [[maybe_unused]] std::uint32_t unused_out,
    VectorMode vector_mode) {
    switch (dst_index) {
        case 0: _mul_bcast_scalar_tile_imm_<ITERATIONS, 0 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 1: _mul_bcast_scalar_tile_imm_<ITERATIONS, 1 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 2: _mul_bcast_scalar_tile_imm_<ITERATIONS, 2 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 3: _mul_bcast_scalar_tile_imm_<ITERATIONS, 3 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 4: _mul_bcast_scalar_tile_imm_<ITERATIONS, 4 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 5: _mul_bcast_scalar_tile_imm_<ITERATIONS, 5 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 6: _mul_bcast_scalar_tile_imm_<ITERATIONS, 6 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        case 7: _mul_bcast_scalar_tile_imm_<ITERATIONS, 7 * DEST_TILE_SIZE_RAW>(vector_mode); return;
        default: break;
    }
    const std::uint32_t base = dst_index * DEST_TILE_SIZE_RAW;
    const bool upper_right = vector_mode == VectorMode::RC || vector_mode == VectorMode::R;
    const bool lower_left = vector_mode == VectorMode::RC || vector_mode == VectorMode::C;
    const bool lower_right = vector_mode == VectorMode::RC;
    _mul_bcast_scalar_face_<ITERATIONS>(base + FACE0_BASE);
    if (upper_right) {
        _mul_bcast_scalar_face_<ITERATIONS>(base + FACE1_BASE);
    }
    if (lower_left) {
        _mul_bcast_scalar_face_<ITERATIONS>(base + FACE2_BASE);
    }
    if (lower_right) {
        _mul_bcast_scalar_face_<ITERATIONS>(base + FACE3_BASE);
    }
}

}  // namespace sfpu
}  // namespace ckernel
