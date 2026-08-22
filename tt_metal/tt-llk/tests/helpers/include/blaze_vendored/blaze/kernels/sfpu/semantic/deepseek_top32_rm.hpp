// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// BUILTIN-BRIDGE LIFT of the deepseek top32 row-major kernel (lane EX,
// 2026-08-21).  Original (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h
//
// Lifted surface = the API-reachable set (llk_math_deepseek_top32_rm.h):
// _top32_rm_init_, _bitonic_top32_of_1024_rm_pre_sorted_{prep_,combine_,
// final_} and their callees (ph3, step_N, merge, rebuild, load16/store16,
// x8 Dst increments).  bitonic_top32_{phases_steps_,load8,store8,ph0,ph1,
// ph2,inc_x4} are unreachable from the API wrapper and are not lifted.
//
// Documented intentional differences (value/state-equivalent, Gn-numbered):
//  G1 index rows ride as plain zero-extended 16-bit companions
//     (InstrModLoadStore::LO16 == MOD0_FMT_UINT16); the lift uses the same
//     mod0 words through file-local raw-mod0 accessors (no typed spelling of
//     load-mod0-6 exists; DataLayout::U16 exists but converts through
//     vUInt16 narrowing types — the raw form keeps the exact word).
//  G2 the ADDR_MOD_6-fused Dst advance (dest.incr=16 on the last index
//     store of store16<alt_addr_mod=true>) is spelled as the same store with
//     the no-increment address mode followed by typed dst_reg increments
//     (see G8 for the word shape).  SFPSTORE.md applies the address-mod
//     after the store, and no access intervenes, so the final RWC state is
//     identical; the lift therefore needs no ADDR_MOD_6 programming and
//     _top32_rm_init_'s addrmod configuration is dropped (G6).
//  G3 the ArgMin direction wrap (TTI_SFPCONFIG(0x104, 0xF, 1) — EXCHANGE_
//     SRCB_SRCC + ENABLE_DEST_INDEX) is the same imm-form LaneConfig
//     vocabulary gap as the parent kit's toggle: kept as the raw imm word
//     via a file-local set_lane_config_imm, INCLUDING the original's two
//     SFPNOPs (they guard config-to-SFPSWAP consumption; the compiler does
//     not model raw config words, so the hand spacing is retained verbatim).
//  G4 the per-instance alternating-direction sections (SFPLOADI 0x0104 +
//     SFPCONFIG(mask, 0xF, 8) lane-mask form) are bridged with the VALUE
//     form: LaneConfig is written in every lane with a tile-id-predicated
//     vector (0x104 on the masked columns, 0x004 elsewhere).  The masks
//     0x4444/0x5050/0x5500 select exactly the columns with (col&1)/(col&2)/
//     (col&4) set, so the predicate is a single AND-compare.  The lane-mask
//     form writes only the selected instances and the others KEEP 0x004 (set
//     earlier); the value form writes the same total state explicitly.
//     These sections run with no live vector state (everything is in Dst),
//     so the value-form's LReg0 staging is pressure-feasible here.
//  G5 runtime (dist0, dist1) load/store distances become template constants
//     (every call site passes literals); TT_-composed words become TTI-class
//     immediate words with identical fields.
//  G6 _top32_rm_init_: the original programs LaneConfig via
//     _sfpu_load_config32_(0xF,0,0x4) (value form through LREG0) and
//     configures ADDR_MOD_6.  The lift sets the same LaneConfig with the
//     imm-form toggle and drops the addrmod program (unused per G2).
//  G7 SETC16 dest-offset rebases (set_dst_write_addr_offset) are protocol,
//     not SFPU math — reused from the original header by identity (the
//     census's established TT_SETC16 retention).
//  G8 typed-increment range gap: __builtin_rvtt_ttincrwc bounds the Dst
//     field to [-8, 7], so the hand kernel's TTINCRWC(0, 8) is spelled as
//     two modeled TTINCRWC(0, 4) words (same final RWC, +1 word/advance).
//  G9 runtime sort directions become template parameters (all API call
//     sites are constexpr); the runtime CFG diamonds defeat IRA coloring.
//  G10 merge/step-N passes are issued as two independent 4-live half passes
//     (disjoint Dst rows) — the 8-live pairing has no transp8 anchor and
//     IRA cannot color it (compile-proven); same instructions, reordered.
//
// Register-state faithfulness: every load16/store16 round trip passes the
// full 8-register bank through Dst; no cross-call LREG state exists in this
// kernel (each section reloads).  The sort math (SFPSWAP mod1=1 under
// ENABLE_DEST_INDEX, dual-bank SFPTRANSP) is bridged 1:1, so NaN/sign-
// magnitude compare behavior is identical by construction.

#include "sfpi.h"
#include "lltt.h"
#include "blaze/kernels/sfpu/semantic/sfpu_bridge.hpp"
// Reuse set_dst_write_addr_offset + SortDir + layout facts (read-only).
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

struct T32Bank {
    sfpi::vFloat v0, v1, v2, v3;
    sfpi::vUInt c0, c1, c2, c3;
};

// G1: zero-extended 16-bit index words (original mod0 InstrModLoadStore::LO16
// == 6 == MOD0_FMT_UINT16 on both load and store).
sfpi_inline sfpi::vUInt t32_load_u16(unsigned addr)
{
    return sfpi::vUInt{__builtin_rvtt_sfpload(addr, sfpi::SFPLOAD_MOD0_FMT_UINT16,
                                              sfpi::SFPLOAD_ADDR_MODE_NOINC)};
}
sfpi_inline void t32_store_u16(const sfpi::vUInt &c, unsigned addr)
{
    __builtin_rvtt_sfpstore(c.get(), addr, sfpi::SFPSTORE_MOD0_FMT_UINT16,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
}

// G3: imm-form LaneConfig write with an arbitrary value (the parent kit's
// documented vocabulary gap, one more value point: 0x104 adds
// EXCHANGE_SRCB_SRCC to invert every SFPSWAP comparison).  The two SFPNOPs
// are the original's own config-consumption spacing, retained verbatim.
template <uint16_t Value>
sfpi_inline void t32_set_lane_config_imm()
{
    TTI_SFPCONFIG(Value, 0xF, 1);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// G4: per-instance alternating swap direction.  Columns with (col & ColBit)
// set get LaneConfig 0x104 (inverted compares), the rest keep 0x004.
template <int ColBit>
sfpi_inline void t32_set_swap_dir_alternating()
{
    static_assert(ColBit == 1 || ColBit == 2 || ColBit == 4);
    sfpi::vInt col = (sfpi::vInt(sfpi::vConstTileId) >> 1) & 7;
    sfpi::vInt cfg = 0x004;
    v_if ((col & ColBit) != 0) {
        cfg = 0x104;
    }
    v_endif;
    __builtin_rvtt_sfpwriteconfig_v(cfg.get(), 15);
}

// Typed Dst-counter advance.  G8 (vocabulary-gap datum): the hand kernel's
// TTI_INCRWC(0, 8, 0, 0) cannot be spelled through the typed increment —
// __builtin_rvtt_ttincrwc bounds the Dst field to [-8, 7] (dst_reg += 4
// would need d=8 and the compiler refuses it), so each 8-row advance is two
// modeled TTINCRWC(0, 4) words (same final RWC, +1 word per advance).
sfpi_inline void t32_inc_dest_8() { sfpi::dst_reg += 2; sfpi::dst_reg += 2; }
sfpi_inline void t32_inc_dest_16() { t32_inc_dest_8(); t32_inc_dest_8(); }

template <uint32_t dist1>
sfpi_inline void t32_load16(T32Bank &b)
{
    constexpr uint32_t idx = 128;  // dst_indices_offset
    b.v0 = load_value(0);
    b.v1 = load_value(4);
    b.v2 = load_value(dist1);
    b.v3 = load_value(dist1 + 4);
    b.c0 = t32_load_u16(idx + 0);
    b.c1 = t32_load_u16(idx + 4);
    b.c2 = t32_load_u16(idx + dist1);
    b.c3 = t32_load_u16(idx + dist1 + 4);
}

template <uint32_t dist1, bool alt_addr_mod>
sfpi_inline void t32_store16(const T32Bank &b)
{
    constexpr uint32_t idx = 128;
    store_value(b.v0, 0);
    store_value(b.v1, 4);
    store_value(b.v2, dist1);
    store_value(b.v3, dist1 + 4);
    t32_store_u16(b.c0, idx + 0);
    t32_store_u16(b.c1, idx + 4);
    t32_store_u16(b.c2, idx + dist1);
    t32_store_u16(b.c3, idx + dist1 + 4);
    if constexpr (alt_addr_mod) {
        t32_inc_dest_16();  // G2: replaces the ADDR_MOD_6-fused dest.incr=16
    }
}

// G9: the originals take `dir` as a runtime bool; every API-reachable call
// site derives it from constexpr values, and the runtime CFG diamonds around
// the transp8-pinned banks are exactly what IRA fails to color
// (lreg-pressure-exceeded on the straight-lined loops) — so the lift takes
// the direction as a template parameter and unrolls the small direction-
// alternating loops.  Same instruction stream, no runtime branches.
template <bool dir>
sfpi_inline void t32_ph3_st4_to_1(T32Bank &b)
{
    if constexpr (dir == static_cast<bool>(SortDir::ArgMin)) {
        t32_set_lane_config_imm<0x104>();  // G3: invert SWAP max/min
    }

    indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);  // Step 4
    indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);  // Step 3
    indexed_swap<1>(b.v2, b.v3, b.c2, b.c3);
    transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);  // Step 4
    indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);  // Step 3
    indexed_swap<1>(b.v2, b.v3, b.c2, b.c3);
    transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);

    if constexpr (dir == static_cast<bool>(SortDir::ArgMin)) {
        t32_set_lane_config_imm<0x004>();  // G3: restore
    }
}

template <bool dir>
sfpi_inline void t32_step_N(T32Bank &b)
{
    if constexpr (dir == static_cast<bool>(SortDir::ArgMax)) {
        indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);
        indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
    } else {
        indexed_swap<1>(b.v2, b.v0, b.c2, b.c0);
        indexed_swap<1>(b.v3, b.v1, b.c3, b.c1);
    }
}

// G10: a merge pass (load16 / step_N / store16) holds 8 live vectors whose
// only register constraint is the indexed-swap value/companion pairing; with
// no transp8 hard-anchor in the pass, IRA fails to discover a valid pairing
// and spills (compile-proven).  The pass's two swap pairs are independent
// and their Dst rows disjoint, so the lift issues them as two 4-live half
// passes — the same instructions on the same rows, reordered.
template <uint32_t dist, bool dir, bool alt_addr_mod>
sfpi_inline void t32_step_pass()
{
    constexpr uint32_t idx = 128;  // dst_indices_offset
    {
        sfpi::vFloat va = load_value(0);
        sfpi::vFloat vb = load_value(dist);
        sfpi::vUInt ca = t32_load_u16(idx + 0);
        sfpi::vUInt cb = t32_load_u16(idx + dist);
        if constexpr (dir == static_cast<bool>(SortDir::ArgMax)) {
            indexed_swap<1>(va, vb, ca, cb);
        } else {
            indexed_swap<1>(vb, va, cb, ca);
        }
        store_value(va, 0);
        store_value(vb, dist);
        t32_store_u16(ca, idx + 0);
        t32_store_u16(cb, idx + dist);
    }
    {
        sfpi::vFloat va = load_value(4);
        sfpi::vFloat vb = load_value(dist + 4);
        sfpi::vUInt ca = t32_load_u16(idx + 4);
        sfpi::vUInt cb = t32_load_u16(idx + dist + 4);
        if constexpr (dir == static_cast<bool>(SortDir::ArgMax)) {
            indexed_swap<1>(va, vb, ca, cb);
        } else {
            indexed_swap<1>(vb, va, cb, ca);
        }
        store_value(va, 4);
        store_value(vb, dist + 4);
        t32_store_u16(ca, idx + 4);
        t32_store_u16(cb, idx + dist + 4);
    }
    if constexpr (alt_addr_mod) {
        t32_inc_dest_16();  // G2
    }
}

template <uint32_t dist, bool dir>
sfpi_inline void t32_merge()
{
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 4; d++) {
        t32_step_pass<dist, dir, false>();
        t32_inc_dest_8();
    }
}

// One direction-alternating pass (load16 / ph3 / store16), G9-unrolled.
template <bool dir>
sfpi_inline void t32_ph3_pass()
{
    T32Bank b{};
    t32_load16<8>(b);
    t32_ph3_st4_to_1<dir>(b);
    t32_store16<8, true>(b);
}
template <bool dir>
sfpi_inline void t32_step5_pass()
{
    t32_step_pass<16, dir, false>();
    t32_inc_dest_8();
    t32_inc_dest_16();
}

template <bool idir, bool skip_second>
sfpi_inline void t32_rebuild()
{
    // Step 5
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    t32_step5_pass<idir>();
    if constexpr (!skip_second) {
        t32_step5_pass<!idir>();
    }
    // Steps 4 to 1
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    t32_ph3_pass<idir>();
    t32_ph3_pass<idir>();
    if constexpr (!skip_second) {
        t32_ph3_pass<!idir>();
        t32_ph3_pass<!idir>();
    }
}

// ---- API-reachable entry points -------------------------------------------

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min>
inline void _semantic_bitonic_top32_of_1024_rm_pre_sorted_prep_(std::uint32_t dst_index)
{
    static_assert(!is_fp32_dest_acc_en, "16-bit index path lifted");
    constexpr std::uint32_t odd_col_offset = 2;
    constexpr bool decreasing = false;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    /// Step 1: build len-32 bitonic sequences from the pre-sorted data
    /// (G9: the d-loop's dir alternation 0,1,0,1 unrolled per column).
    for (int col = 0; col < 2; col++) {
        lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
        t32_ph3_pass<decreasing>();
        t32_ph3_pass<!decreasing>();
        t32_ph3_pass<decreasing>();
        t32_ph3_pass<!decreasing>();
        t32_rebuild<decreasing, /* skip_second */ false>();
        set_dst_write_addr_offset(tile_offset + odd_col_offset);  // G7
    }
    set_dst_write_addr_offset(tile_offset);

    /// Step 2: merge and rebuild F0/F1 with F2/F3 (col dir = top_min, !top_min).
    t32_merge<32, decreasing>();
    t32_rebuild<top_min, /* skip_second */ true>();
    set_dst_write_addr_offset(tile_offset + odd_col_offset);
    t32_merge<32, decreasing>();
    t32_rebuild<!top_min, /* skip_second */ true>();
    set_dst_write_addr_offset(tile_offset + odd_col_offset);
    set_dst_write_addr_offset(tile_offset);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _semantic_bitonic_top32_of_1024_rm_pre_sorted_combine_(std::uint32_t dst_index)
{
    static_assert(!is_fp32_dest_acc_en, "16-bit index path lifted");
    constexpr std::uint32_t odd_col_offset = 2;
    constexpr bool decreasing = false;
    constexpr bool increasing = true;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    t32_merge<64, decreasing>();
    t32_rebuild<decreasing, /* skip_second */ true>();
    set_dst_write_addr_offset(tile_offset + odd_col_offset);
    t32_merge<64, decreasing>();
    t32_rebuild<increasing, /* skip_second */ true>();
    set_dst_write_addr_offset(tile_offset);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _semantic_bitonic_top32_of_1024_rm_pre_sorted_final_(std::uint32_t dst_index)
{
    static_assert(!is_fp32_dest_acc_en, "16-bit index path lifted");
    constexpr bool decreasing = false;
    constexpr std::uint32_t odd_col_offset = 2;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    /// Step 1: merge even/odd cols and rebuild, store to odd cols.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 4; d++) {
        t32_step_pass<odd_col_offset, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    t32_set_swap_dir_alternating<1>();  // G4: original 0x4444 lane mask
    for (int d = 0; d < 2; d++) {
        t32_step_pass<16, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        t32_load16<8>(b);
        t32_ph3_st4_to_1<decreasing>(b);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        t32_store16<8, true>(b);
        set_dst_write_addr_offset(tile_offset);
    }

    /// Step 2: shift odd cols right by 1 SFPU instance, store to even cols.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    set_dest_index_tracking<false>();  // TEN-2932: SHFT2 outside the window
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        t32_load16<8>(b);
        ror1_ip(b.v0); ror1_ip(b.v1); ror1_ip(b.v2); ror1_ip(b.v3);
        ror1_ip(b.c0); ror1_ip(b.c1); ror1_ip(b.c2); ror1_ip(b.c3);
        set_dst_write_addr_offset(tile_offset);
        t32_store16<8, true>(b);
    }
    set_dest_index_tracking<true>();

    /// Step 3: merge/rebuild, alternate every 2nd instance.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 4; d++) {
        t32_step_pass<odd_col_offset, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    t32_set_swap_dir_alternating<2>();  // G4: original 0x5050 lane mask
    for (int d = 0; d < 2; d++) {
        t32_step_pass<16, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        t32_load16<8>(b);
        t32_ph3_st4_to_1<decreasing>(b);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        t32_store16<8, true>(b);
        set_dst_write_addr_offset(tile_offset);
    }

    /// Step 4: shift odd cols right by 2 instances, store to even cols.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    set_dest_index_tracking<false>();
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        t32_load16<8>(b);
        for (int i = 0; i < 2; i++) {
            ror1_ip(b.v0); ror1_ip(b.v1); ror1_ip(b.v2); ror1_ip(b.v3);
            ror1_ip(b.c0); ror1_ip(b.c1); ror1_ip(b.c2); ror1_ip(b.c3);
        }
        set_dst_write_addr_offset(tile_offset);
        t32_store16<8, true>(b);
    }
    set_dest_index_tracking<true>();

    /// Step 5: merge/rebuild, alternate every 4th instance.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 4; d++) {
        t32_step_pass<odd_col_offset, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    t32_set_swap_dir_alternating<4>();  // G4: original 0x5500 lane mask
    for (int d = 0; d < 2; d++) {
        t32_step_pass<16, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        t32_load16<8>(b);
        t32_ph3_st4_to_1<decreasing>(b);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        t32_store16<8, true>(b);
        set_dst_write_addr_offset(tile_offset);
    }

    /// Step 6: shift odd cols right by 4 instances, store to even cols.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    set_dest_index_tracking<false>();
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        t32_load16<8>(b);
        for (int i = 0; i < 4; i++) {
            ror1_ip(b.v0); ror1_ip(b.v1); ror1_ip(b.v2); ror1_ip(b.v3);
            ror1_ip(b.c0); ror1_ip(b.c1); ror1_ip(b.c2); ror1_ip(b.c3);
        }
        set_dst_write_addr_offset(tile_offset);
        t32_store16<8, true>(b);
    }
    set_dest_index_tracking<true>();

    /// Step 7: final merge/rebuild, store to even cols.
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 4; d++) {
        t32_step_pass<odd_col_offset, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 2; d++) {
        t32_step_pass<16, decreasing, false>();
        t32_inc_dest_8();
    }
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    for (int d = 0; d < 2; d++) {
        T32Bank b{};
        t32_load16<8>(b);
        t32_ph3_st4_to_1<decreasing>(b);
        t32_store16<8, true>(b);
    }
}

inline void _semantic_top32_rm_init_()
{
    // G6: same LaneConfig end state as _sfpu_load_config32_(0xF, 0, 0x4);
    // the ADDR_MOD_6 program is dropped (the lift never uses it, per G2).
    set_dest_index_tracking<true>();
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
