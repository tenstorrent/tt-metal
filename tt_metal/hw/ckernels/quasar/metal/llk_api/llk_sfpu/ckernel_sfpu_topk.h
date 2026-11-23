// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

// Quasar TopK keeps the SFPSWAP bodies inline. TEN-4690 forbids record-and-execute
// replay (`execute_while_loading=true`), and replaying recorded SFPSWAP sequences
// loses the scheduling behavior this bitonic network depends on. Load/store replay
// would be safe, but keeping each compare block inline makes the correctness contract
// explicit and avoids mixing replayed and non-replayed SFPU state transitions.

// NOTE: The comparator-stable network (STABLE_SORT) is NOT supported on Quasar. The public entry
// points keep the `STABLE_SORT` template parameter for Blackhole call-shape parity, but assert it
// off (`static_assert(!STABLE_SORT, ...)`). Stable TopK runs through the two key modes instead,
// FUSED (packed [bf16 value | u16 index'] keys) and RANK_STAMPED (rank-tagged value words), which
// sort with the unstable network; see the "Stable key modes" section below.

namespace ckernel {
namespace sfpu {

// Sort direction for topk. Defined locally — Quasar's llk_defs.h does not define SortDir.
enum SortDir : bool {
    ArgMax = false,
    ArgMin = true,
};

enum class TopkTieOrder : std::uint8_t {
    Unset,
    Ascending,
    Descending,
};

// SFPSWAP imm12 bit 0 selects the compare: 0 = two's-complement int32, 1 = fp32. TopK values
// are always floats (FP16B/FP32), and their sign-magnitude bit patterns order backwards under
// the int32 compare when both operands are negative, so the plain (unstable) network uses the fp32
// compare. The fp32 compare treats -0 == +0 and never swaps when either operand is NaN. The stable
// key modes below sort int32-ordered keys with the int32 compare instead.
constexpr std::uint32_t TOPK_SWAP_IMM12_FP32 = 0x1;
constexpr std::uint32_t TOPK_SWAP_IMM12_INT32 = 0x0;

// =============================================================================
//  Stable key modes (FUSED, RANK_STAMPED)
// =============================================================================
//
// Both modes are the Blackhole ones (see the Blackhole ckernel_sfpu_topk.h for the full contract):
// the network sorts 32-bit words whose low bits carry a tie-break tag, so equal values come out in
// torch's stable (index-ascending) order.
//   * FUSED: word = [bf16 value | idx XOR (0xFFFF iff (value >= 0) == largest)]; the index tiles are
//     consumed by the fuse and rebuilt by the defuse, and index tracking is OFF.
//   * RANK_STAMPED: word = [value | rank XOR (TAG_MASK iff (value >= 0) == largest)] in the low
//     TAG_BITS bits; the true indices ride index tracking at Dest offset 128.
// The words in Dest (and in any L1 round trip) are bit-identical to Blackhole's.
//
// Blackhole's SFPSWAP orders such words by sign and magnitude. Quasar's SFPSWAP compares only as
// int32 (two's complement) or fp32. The fp32 compare cannot order them: a tagged +-inf is a NaN
// pattern (never swapped) and a tagged zero is a denormal. So the network loads every value word,
// flips it into an int32-ordered key, sorts with the int32 compare, and flips it back before the
// store: key(w) = w XOR (w < 0 ? 0x7FFFFFFF : 0). For w >= 0 the key is w; for a negative w with
// magnitude bits m it is -1 - m, so int32 order on keys is sign-magnitude order on words (-0 just
// below +0, as on Blackhole). The flip is an involution and keeps the sign bit, so the same four
// ALU instructions convert both ways, with no constant register and no load (TEN-2932 safe).
constexpr std::uint32_t TOPK_INDEX_DEST_OFFSET = 128;  // index tiles 2 and 3: two tiles of 64 Dest rows

// Programmable constant LREGs the key modes use (written through SFPCONFIG). LREG11 is sfpi's -1.
constexpr std::uint32_t TOPK_LREG_MASK = 12;       // fused: 0x0000FFFF; rank-stamped: the tag mask
constexpr std::uint32_t TOPK_LREG_RANK_SPAN = 13;  // rank-stamped merge: 2K - 1
constexpr std::uint32_t TOPK_LREG_TAG_CLEAR = 14;  // rank-stamped: ~tag mask
constexpr std::uint32_t TOPK_LREG_TILE_ID = 15;    // vConstTileId: 2 * lane

// Low half of a 32-bit Dest word: the u16 index half of a fused key and its tie complement.
constexpr std::uint32_t TOPK_LO16_MASK = 0x0000FFFF;
// The index_store_mode the Compute API passes to calculate_topk_defuse (Blackhole's SFPSTORE
// mode 9, a low-to-high half move for a UInt16 pack). UInt16 does not exist on Quasar; there the
// defused index stays a plain integer in the 32-bit word, packed as Int32.
constexpr std::uint32_t TOPK_SFPSTORE_MODE_PACK_UINT16 = 9;

constexpr std::uint32_t TOPK_SETCC_NEGATIVE = 0x0;  // CC <= (src_c < 0); imm12 bit 11 = 0: as int32
constexpr std::uint32_t TOPK_SETCC_ZERO = 0x6;      // CC <= (src_c == 0)
constexpr std::uint32_t TOPK_SFPLOADI_UINT16 = 0x2;
constexpr std::uint32_t TOPK_SFPLOADI_LO16_ONLY = 0xA;   // write LREG[15:0], keep LREG[31:16]
constexpr std::uint32_t TOPK_SFPSHFT_C_BY_IMM = 0x5;     // dest = src_c << imm (logical; imm < 0 shifts right)
constexpr std::uint32_t TOPK_SFPIADD_IMM_KEEP_CC = 0x5;  // dest = src_c + imm, CC untouched

// SFPSWAP compare for the mode: int32 on flipped keys for the key modes, fp32 on plain values.
template <bool FUSED, bool RANK_STAMPED>
constexpr std::uint32_t topk_swap_imm12() {
    return (FUSED || RANK_STAMPED) ? TOPK_SWAP_IMM12_INT32 : TOPK_SWAP_IMM12_FP32;
}

// Rank-stamped keys carry a TAG_BITS-wide rank tag in the value word's low bits: 16 for bf16 values
// (the whole low half is free), 6..15 for fp32 keys whose low mantissa bits are known to be zero.
template <std::uint32_t TAG_BITS>
constexpr std::uint32_t topk_tag_mask() {
    static_assert(TAG_BITS >= 6 && TAG_BITS <= 16, "rank tag field must be 6..16 bits (a 64-rank chain needs 6)");
    return (1u << TAG_BITS) - 1u;
}

// All lanes enabled, CC result true.
inline void topk_lanes_on() { TTI_SFPENCC(0 /*imm12*/, 0 /*instr_mod1: result = 1, keep enable*/); }

// One programmable constant LREG. The SFPCONFIG path is lane-predicated and clobbers LREG0, so
// enable every lane first and program before any load.
inline void topk_load_lreg_constant(const std::uint32_t lreg, const std::uint32_t value) {
    topk_lanes_on();
    ckernel::math::_sfpu_load_config32_(lreg, value >> 16, value & 0xFFFF);
}

// TARGET ^= TOPK_LREG_MASK in the lanes whose word (SIGN_LREG) has a sign that matches the kept
// extreme: (word >= 0) == LARGEST. LARGEST complements every lane, then un-complements the negative
// ones (one more instruction than an inverted CC; ttsim has no SFPSETCC CC-invert mode).
template <bool LARGEST, std::uint32_t SIGN_LREG, std::uint32_t TARGET>
inline void topk_complement_tag() {
    if constexpr (LARGEST) {
        TTI_SFPXOR(TOPK_LREG_MASK, TARGET);
    }
    TTI_SFPSETCC(0 /*imm12: as int32*/, SIGN_LREG, TOPK_SETCC_NEGATIVE);
    TTI_SFPXOR(TOPK_LREG_MASK, TARGET);
    topk_lanes_on();
}

// key(w) = w XOR (w < 0 ? 0x7FFFFFFF : 0), in place; its own inverse (see the key modes above).
template <std::uint32_t LREG>
inline void topk_flip_key() {
    TTI_SFPSETCC(0 /*imm12: as int32*/, LREG, TOPK_SETCC_NEGATIVE);
    TTI_SFPNOT(LREG, LREG);
    TTI_SFPSETSGN(1 /*imm12: sign = 1*/, LREG, LREG, 1 /*instr_mod1: sign from imm12*/);
    topk_lanes_on();
}

template <bool FLIP>
inline void topk_flip_keys_0_1() {
    if constexpr (FLIP) {
        topk_flip_key<p_sfpu::LREG0>();
        topk_flip_key<p_sfpu::LREG1>();
    }
}

template <bool FLIP>
inline void topk_flip_keys_0_3() {
    if constexpr (FLIP) {
        topk_flip_key<p_sfpu::LREG0>();
        topk_flip_key<p_sfpu::LREG1>();
        topk_flip_key<p_sfpu::LREG2>();
        topk_flip_key<p_sfpu::LREG3>();
    }
}

// Set the per-TRISC dest section base register for the math TRISC.
// Quasar has separate SEC0..SEC3 registers (one per TRISC); this implementation
// runs the SFPU TopK network on the math TRISC, so SFPLOAD/SFPSTORE effective
// addresses use SEC1 + dest_counter + dest_reg_addr. A future TRISC3 split would
// need to program that TRISC's section base instead.
inline void set_dst_write_addr(std::uint32_t addr) {
    std::uint32_t dst_index = addr + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<ckernel::TRISC_ID>(dst_index);
}

// Advance the dest RWC counter by `inc` rows in groups of 8.
// `cr=true` issues an additional carriage-return bit to clear the column counter.
// Quasar TTI_INCRWC arg order is (rwc_cr, rwc_a, rwc_b, rwc_d) — dest is the 4th argument.
inline void bitonic_topk_inc_x8_dest(std::uint32_t inc, bool cr) {
    std::uint32_t inc_grp8 = inc >> 3;
    if (cr) {
        for (std::uint32_t i = 0; i < inc_grp8; i++) {
            TTI_INCRWC(0b100, 0, 0, 8);
        }
    } else {
        for (std::uint32_t i = 0; i < inc_grp8; i++) {
            TTI_INCRWC(0, 0, 0, 8);
        }
    }
}

/**
 * @brief Set LaneConfig bit [2] (ENABLE_DEST_INDEX) so SFPSWAP tracks indices alongside values.
 *
 * @note The bit persists on Quasar, so set it once before the topk execute stages, not per call.
 */
inline void init_topk() {
    // Write 0x4 to LaneConfig (config_dest=0xF) to set bit [2] = ENABLE_DEST_INDEX.
    // With this bit set, SFPSWAP performs argmin/argmax: when it conditionally
    // swaps LREG[VC] <-> LREG[VD], it also swaps LREG[4 + (VC&3)] <-> LREG[4 + (VD&3)]
    // in lockstep — letting topk track input indices alongside the values being sorted.
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x4);
    // SFPCONFIG is a 2-cycle op; per Quasar errata TEN-4581 ("any 2-cycle op
    // followed by SFPSWAP") at least 1 SFPNOP must separate it from the first
    // SFPSWAP in the TopK body. Keep two NOPs to match the other SFPCONFIG
    // sites below and the Blackhole-style spacing used during bring-up.
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
}

// Program ADDR_MOD_6 with dest.incr=32: the final index store (store16 alt_addr_mod path) uses it
// to auto-advance Dest by 32 rows. Not set by the default SFPU init.
inline void init_topk_addr_mod() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
    }
        .set(ADDR_MOD_6);
}

/**
 * @brief Fused-key init: index tracking OFF (the packed key carries the index) and LREG12 = 0xFFFF.
 *
 * @note Written as an explicit clear so a preceding tracked-mode topk cannot leak the bit in.
 */
inline void init_topk_fused() {
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x0);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
    topk_load_lreg_constant(TOPK_LREG_MASK, TOPK_LO16_MASK);
}

/**
 * @brief Rank-stamped init: index tracking ON (the true indices ride the tracked swaps) and
 *        LREG12 = the tag mask. The merge programs its other constants at each entry.
 *
 * @tparam TAG_BITS: Rank tag field width in the value word's low bits, values = <6..16>
 */
template <std::uint32_t TAG_BITS>
inline void init_topk_rank_stamped() {
    init_topk();
    topk_load_lreg_constant(TOPK_LREG_MASK, topk_tag_mask<TAG_BITS>());
}

/**
 * @brief Configure topk-specific SFPU state: ADDR_MOD_6 (@ref init_topk_addr_mod) and the mode's
 *        LaneConfig ENABLE_DEST_INDEX bit and constants.
 *
 * @tparam FUSED: Fused-key mode (@ref init_topk_fused), values = <true/false>
 * @tparam RANK_STAMPED: Rank-stamped mode (@ref init_topk_rank_stamped), values = <true/false>
 * @tparam TAG_BITS: Rank tag field width, rank-stamped mode only, values = <6..16>
 * @note Run after the base SFPU init and before the topk execute stages.
 */
template <bool APPROXIMATE, bool FUSED = false, bool RANK_STAMPED = false, std::uint32_t TAG_BITS = 16>
inline void topk_init() {
    static_assert(!(FUSED && RANK_STAMPED), "fused and rank-stamped modes are mutually exclusive");
    static_assert(RANK_STAMPED || TAG_BITS == 16, "TAG_BITS applies to the rank-stamped mode only");
    init_topk_addr_mod();
    if constexpr (FUSED) {
        init_topk_fused();
    } else if constexpr (RANK_STAMPED) {
        init_topk_rank_stamped<TAG_BITS>();
    } else {
        init_topk();
    }
}

// Point the SFPU at Dest row `addr` of the slab with the RWC counter at 0.
inline void topk_sweep_start(const std::uint32_t addr) {
    set_dst_write_addr(addr);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
}

// Next 32-lane vector: 4 Dest rows of one column parity, 2 Dest address units on.
inline void topk_sweep_next_vector() { TTI_INCRWC(0, 0, 0, 2); }

/**
 * @brief Fuse a freshly loaded 2-tile slab into packed [bf16 value | u16 index'] keys, in place.
 *
 * Value tiles at Dest tiles 0,1 as exact [bf16 | 0x0000] words, index tiles at tiles 2,3 holding
 * the index as an integer in the low 16 bits of each 32-bit word (Int32 on Quasar). The index is
 * complemented iff (value >= 0) == LARGEST, so the lower index wins a tie in the requested GLOBAL
 * direction. -0.0 is folded into +0.0 on the way (torch ties them; Quasar Dest keeps -0.0, where
 * the Blackhole datacopy path does not deliver it). The index tiles are dead afterwards.
 *
 * @tparam LARGEST: The global sort order, values = <true: largest first / false: smallest first>
 * @note Needs 32-bit Dest and @ref topk_init<..., true> (fused). Run once per fresh slab, before
 *       its local sort; @ref calculate_topk_defuse undoes it on the final tiles.
 */
template <bool LARGEST>
inline void _topk_fuse_tile_() {
    topk_load_lreg_constant(TOPK_LREG_MASK, TOPK_LO16_MASK);
    topk_sweep_start(0);
    for (int vector = 0; vector < 64; vector++) {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);                       // [bf16|0]
        TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET);  // index
        // -0.0 -> +0.0: (w << 1) == 0 exactly for the two zero encodings.
        TTI_SFPSHFT(1 /*imm12: shift*/, p_sfpu::LREG0, p_sfpu::LREG2, TOPK_SFPSHFT_C_BY_IMM);
        TTI_SFPSETCC(0 /*imm12*/, p_sfpu::LREG2, TOPK_SETCC_ZERO);
        TTI_SFPMOV(p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*instr_mod1: plain copy*/);
        topk_lanes_on();
        TTI_SFPAND(TOPK_LREG_MASK, p_sfpu::LREG1);  // the u16 index
        topk_complement_tag<LARGEST, p_sfpu::LREG0, p_sfpu::LREG1>();
        TTI_SFPOR(p_sfpu::LREG1, p_sfpu::LREG0);  // packed key
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
        topk_sweep_next_vector();
    }
    topk_sweep_start(0);
}

/**
 * @brief Split num_tiles packed key tiles (Dest tiles 0 onward) back into [bf16 | 0x0000] value
 *        words, in place, and integer index words at tiles 2 onward.
 *
 * @tparam LARGEST: The global sort order the fuse used, values = <true/false>
 * @tparam index_store_mode: INT32, or the Compute API's TOPK_SFPSTORE_MODE_PACK_UINT16, which on
 *         Quasar also stores the index as a plain integer (packed as Int32; no UInt16 on Quasar)
 * @param num_tiles: Packed key tiles to split, values = <1, 2>
 * @note Run once on the final output tiles.
 */
template <bool LARGEST, std::uint32_t index_store_mode = p_sfpu::sfpmem::INT32>
inline void _topk_defuse_tile_(const int num_tiles) {
    static_assert(
        index_store_mode == p_sfpu::sfpmem::INT32 || index_store_mode == TOPK_SFPSTORE_MODE_PACK_UINT16,
        "the Quasar defuse stores the index as an integer word");
    topk_load_lreg_constant(TOPK_LREG_MASK, TOPK_LO16_MASK);
    topk_sweep_start(0);
    const int vectors = 32 * num_tiles;
    for (int vector = 0; vector < vectors; vector++) {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);  // [bf16 | idx']
        // The sign test runs on the packed word, whose bit 31 is the value's sign, so it selects the
        // lanes the fuse complemented: the complement undoes itself.
        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0 /*instr_mod1: plain copy*/);
        TTI_SFPAND(TOPK_LREG_MASK, p_sfpu::LREG1);
        topk_complement_tag<LARGEST, p_sfpu::LREG0, p_sfpu::LREG1>();
        TTI_SFPLOADI(p_sfpu::LREG0, TOPK_SFPLOADI_LO16_ONLY, 0);  // [bf16 | 0x0000]
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET);
        topk_sweep_next_vector();
    }
    topk_sweep_start(0);
}

/**
 * @brief Stamp one value tile (Dest tile 0 or 1) with sign-conditioned rank tags rank_base + [0, 32)
 *        in a TAG_BITS-wide field, in place: word = [value | rank XOR (TAG_MASK iff
 *        (value >= 0) == LARGEST)]. Clears the stale field first and folds -0.0 into +0.0.
 *
 * Within a 16-row face consecutive Dest rows are consecutive sequence positions; rows 16..31 of the
 * transposed tile hold positions 0..15 of the other column half. An SFPLOAD vector covers 4 rows,
 * lane group j = row j of the group (j = vConstTileId >> 4).
 *
 * @tparam LARGEST: The global sort order, values = <true/false>
 * @tparam TAG_BITS: Rank tag field width, values = <6..16>
 * @param dst_tile_index: The slab's value tile, values = <0, 1>
 * @param rank_base: First rank; a multiple of 32 with rank_base + 31 inside the tag field
 * @note Clobbers LREG0..2. Needs 32-bit Dest and the rank-stamped @ref topk_init.
 */
template <bool LARGEST, std::uint32_t TAG_BITS = 16>
inline void _topk_stamp_tile_rank_range_(const std::uint32_t dst_tile_index, const std::uint32_t rank_base) {
    constexpr std::uint32_t TAG_MASK = topk_tag_mask<TAG_BITS>();
    LLK_ASSERT(dst_tile_index <= 1, "stamp_tile_rank_range expects dst tile 0 or 1");
    LLK_ASSERT((rank_base & 31u) == 0u, "stamp_tile_rank_range expects rank_base to be a multiple of 32");
    LLK_ASSERT(rank_base <= TAG_MASK - 31u, "stamp_tile_rank_range expects rank_base + 31 to fit in the tag field");
    topk_load_lreg_constant(TOPK_LREG_MASK, TAG_MASK);
    if constexpr (TAG_BITS != 16) {
        topk_load_lreg_constant(TOPK_LREG_TAG_CLEAR, ~TAG_MASK);  // a narrow field is cleared by SFPAND
    }
    topk_sweep_start(dst_tile_index * 64);  // one 32-bit tile = 64 Dest address units

    // Per-lane rank: the lane group, plus the runtime base (disjoint bits: the iota is below 32).
    TTI_SFPSHFT((-4) & 0xFFF /*imm12: shift right 4*/, TOPK_LREG_TILE_ID, p_sfpu::LREG2, TOPK_SFPSHFT_C_BY_IMM);
    if (rank_base != 0) {
        TT_SFPLOADI(p_sfpu::LREG1, TOPK_SFPLOADI_UINT16, rank_base);
        TTI_SFPOR(p_sfpu::LREG1, p_sfpu::LREG2);
    }

    for (int group = 0; group < 16; group++) {        // 4-row groups across the tile
        for (int parity = 0; parity < 2; parity++) {  // even / odd Dest columns
            TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
            if constexpr (TAG_BITS == 16) {
                TTI_SFPLOADI(p_sfpu::LREG0, TOPK_SFPLOADI_LO16_ONLY, 0);  // [bf16 | 0]
            } else {
                TTI_SFPAND(TOPK_LREG_TAG_CLEAR, p_sfpu::LREG0);  // clear the stale tag field only
            }
            // -0.0 -> +0.0: (w << 1) == 0 exactly for the two zero encodings.
            TTI_SFPSHFT(1 /*imm12: shift*/, p_sfpu::LREG0, p_sfpu::LREG1, TOPK_SFPSHFT_C_BY_IMM);
            TTI_SFPSETCC(0 /*imm12*/, p_sfpu::LREG1, TOPK_SETCC_ZERO);
            TTI_SFPMOV(p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*instr_mod1: plain copy*/);
            topk_lanes_on();
            // Tag with the sequence position, sign-conditioned.
            TTI_SFPOR(p_sfpu::LREG2, p_sfpu::LREG0);
            topk_complement_tag<LARGEST, p_sfpu::LREG0, p_sfpu::LREG0>();
            TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
            topk_sweep_next_vector();
        }
        // Next 4-row group: +4 within a face; crossing into the paired face (the same sequence
        // positions, other column half) rewinds by 12 instead.
        if ((group & 7) == 3) {
            TTI_SFPIADD((-12) & 0xFFF /*imm12*/, p_sfpu::LREG2, p_sfpu::LREG2, TOPK_SFPIADD_IMM_KEEP_CC);
        } else {
            TTI_SFPIADD(4 /*imm12*/, p_sfpu::LREG2, p_sfpu::LREG2, TOPK_SFPIADD_IMM_KEEP_CC);
        }
    }
    topk_sweep_start(0);
}

/**
 * @brief Stamp a fresh 2-tile slab with its sequence positions: rank 0..31 on tile 0, 32..63 on
 *        tile 1 (see @ref _topk_stamp_tile_rank_range_).
 */
template <bool LARGEST, std::uint32_t TAG_BITS = 16>
inline void _topk_stamp_local_positions_() {
    _topk_stamp_tile_rank_range_<LARGEST, TAG_BITS>(0 /*dst_tile_index*/, 0 /*rank_base*/);
    _topk_stamp_tile_rank_range_<LARGEST, TAG_BITS>(1 /*dst_tile_index*/, 32 /*rank_base*/);
}

/**
 * @brief Clear the low TAG_BITS bits (stale rank tags) of one value tile, leaving exact value words
 *        for the pack.
 *
 * @param dst_tile_index: The value tile, values = <0, 1>
 * @note Run on MATH while Dest is acquired, after the last network call. Leaves LREG12 holding the
 *       strip mask; every stamp / merge / fuse entry reprograms it.
 */
template <std::uint32_t TAG_BITS = 16>
inline void _topk_strip_rank_tags_(const std::uint32_t dst_tile_index) {
    LLK_ASSERT(dst_tile_index <= 1, "strip_rank_tags expects dst tile 0 or 1");
    topk_load_lreg_constant(TOPK_LREG_MASK, ~topk_tag_mask<TAG_BITS>());
    topk_sweep_start(dst_tile_index * 64);
    for (int vector = 0; vector < 32; vector++) {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
        TTI_SFPAND(TOPK_LREG_MASK, p_sfpu::LREG0);
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 0);
        topk_sweep_next_vector();
    }
    topk_sweep_start(0);
}

// Mode-combination rules shared by the three network entry points (local sort, merge, rebuild).
template <bool is_fp32_dest_acc_en, bool STABLE_SORT, bool FUSED, bool RANK_STAMPED>
inline void topk_validate_mode_combo() {
    static_assert(!STABLE_SORT, "Stable TopK is not supported by the Quasar bitonic TopK path");
    static_assert(!(RANK_STAMPED && FUSED), "rank-stamped and fused-key modes are mutually exclusive");
    static_assert(!FUSED || is_fp32_dest_acc_en, "fused packed keys require 32-bit Dest");
    static_assert(!RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped tagged keys require 32-bit Dest");
}

// Fused-key sweeps. LARGEST is the op's GLOBAL sort order; fuse runs once per fresh 2-tile slab,
// defuse once on the final output tiles.
template <bool APPROXIMATION_MODE, bool largest>
inline void calculate_topk_fuse() {
    _topk_fuse_tile_<largest>();
}

template <bool APPROXIMATION_MODE, bool largest, std::uint32_t index_store_mode = p_sfpu::sfpmem::INT32>
inline void calculate_topk_defuse(const std::uint32_t num_tiles) {
    _topk_defuse_tile_<largest, index_store_mode>(static_cast<int>(num_tiles));
}

// Rank-stamped stamp sweeps: the whole fresh slab, or one tile with a runtime rank base (the k > 32
// insertion cascade).
template <bool APPROXIMATION_MODE, bool largest, std::uint32_t TAG_BITS = 16>
inline void calculate_topk_stamp_local_positions() {
    _topk_stamp_local_positions_<largest, TAG_BITS>();
}

template <bool APPROXIMATION_MODE, bool largest, std::uint32_t TAG_BITS = 16>
inline void calculate_topk_stamp_tile_rank_range(const std::uint32_t dst_tile_index, const std::uint32_t rank_base) {
    _topk_stamp_tile_rank_range_<largest, TAG_BITS>(dst_tile_index, rank_base);
}

// The comparator-stable network's -0.0 fold. That network does not exist on Quasar (STABLE_SORT is
// asserted off) and the key modes fold -0.0 in their own stamp / fuse, so there is nothing to do.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_topk_canonicalize_negzero() {}

// Value / index SFPLOAD-SFPSTORE formats. TopK indices are non-negative Int16 values in the
// 16-bit Dest harness: use the matching SFPU memory mode instead of transporting index bits through
// FP16B. Plain values use an explicit FP16B / FP32 mode rather than sfpmem::DEFAULT, so they do not
// depend on the ALU format left by the index stage. The key modes move raw 32-bit words (a float
// store would flush a tagged zero, which is a denormal pattern).
template <bool is_fp32_dest_acc_en, bool FUSED, bool RANK_STAMPED>
constexpr std::uint32_t topk_value_sfpmem() {
    if constexpr (FUSED || RANK_STAMPED) {
        return p_sfpu::sfpmem::INT32;
    } else {
        return is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    }
}

template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t topk_index_sfpmem() {
    return is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;
}

// Load 8 lanes (one value LREG pair + one index LREG pair) from Dest at runtime offsets.
// Values land in LREG0,1; indices (offset by TOPK_INDEX_DEST_OFFSET) land in LREG4,5. The fused
// mode has no index tiles; the key modes flip the value words into int32-ordered keys, unless the
// caller has more to do to the words first (FLIP_KEYS = false; the rank-stamped merge re-stamps).
template <
    bool is_fp32_dest_acc_en,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    bool FLIP_KEYS = (FUSED || RANK_STAMPED)>
inline void bitonic_topk_load8(std::uint32_t offset, std::uint32_t dist) {
    constexpr std::uint32_t instr_mod_value = topk_value_sfpmem<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>();
    constexpr std::uint32_t instr_mod_index = topk_index_sfpmem<is_fp32_dest_acc_en>();

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices (paired with LREG0,1)
    if constexpr (!FUSED) {
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + ld_offset);
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + ld_offset + dist);
    }

    // After the index loads: the flip is ALU-only, so the tracked indices are safe (TEN-2932).
    topk_flip_keys_0_1<FLIP_KEYS>();
}

// Store 8 lanes (the same LREGs that bitonic_topk_load8 fills) back into Dest. Mirrors load8: the
// key modes flip the keys back to words first.
template <bool is_fp32_dest_acc_en, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_store8(std::uint32_t offset, std::uint32_t dist) {
    constexpr std::uint32_t instr_mod_value = topk_value_sfpmem<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>();
    constexpr std::uint32_t instr_mod_index = topk_index_sfpmem<is_fp32_dest_acc_en>();

    topk_flip_keys_0_1<FUSED || RANK_STAMPED>();

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices
    if constexpr (!FUSED) {
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + ld_offset);
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + ld_offset + dist);
    }
}

// Load 16 lanes (LREG0..3 values + LREG4..7 indices) from Dest at strided offsets
// (0, dist0, dist1, dist1+dist0). The (dist0,dist1)==(4,8) call site is the hot path
// used by phases 5/6 — its addresses are constexpr so we hand them to TTI_SFPLOAD.
// The key modes flip the value words into int32-ordered keys after the loads.
template <bool is_fp32_dest_acc_en, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_load16(std::uint32_t dist0, std::uint32_t dist1) {
    constexpr std::uint32_t instr_mod_value = topk_value_sfpmem<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>();
    constexpr std::uint32_t instr_mod_index = topk_index_sfpmem<is_fp32_dest_acc_en>();

    // Values
    TTI_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 0, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, 0, 12);
    } else {
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, dist0);
        TT_SFPLOAD(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 0, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, 0, dist1 + dist0);
    }

    // Indices (paired with LREG0..3)
    if constexpr (!FUSED) {
        TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 0);
        if ((dist0 == 4) && (dist1 == 8)) {
            TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 4);
            TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 8);
            TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 12);
        } else {
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + dist0);
            TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + dist1);
            TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + dist1 + dist0);
        }
    }

    // After the index loads: the flip is ALU-only, so the tracked indices are safe (TEN-2932).
    topk_flip_keys_0_3<FUSED || RANK_STAMPED>();
}

// Store 16 lanes (the same LREGs that bitonic_topk_load16 fills) back into Dest, flipping the key
// modes' keys back to words first. When alt_addr_mod=true, the FINAL store (the LREG7 index, or the
// LREG3 value in the fused mode, which has no index stores) uses ADDR_MOD_6 instead of ADDR_MOD_7 —
// ADDR_MOD_6 has dest.incr=32, so this auto-advances Dest by 32 rows after a 16-element block.
template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_store16(std::uint32_t dist0, std::uint32_t dist1) {
    constexpr std::uint32_t instr_mod_value = topk_value_sfpmem<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>();
    constexpr std::uint32_t instr_mod_index = topk_index_sfpmem<is_fp32_dest_acc_en>();
    constexpr std::uint32_t last_value_addr_mod = (FUSED && alt_addr_mod) ? ADDR_MOD_6 : ADDR_MOD_7;

    topk_flip_keys_0_3<FUSED || RANK_STAMPED>();

    // Values
    TTI_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 0, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, instr_mod_value, last_value_addr_mod, 0, 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, dist0);
        TT_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 0, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, instr_mod_value, last_value_addr_mod, 0, dist1 + dist0);
    }

    // Indices — last store optionally swaps to ADDR_MOD_6 for the auto-advance.
    if constexpr (!FUSED) {
        TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 0);
        if ((dist0 == 4) && (dist1 == 8)) {
            TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 4);
            TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 8);
            TTI_SFPSTORE(
                p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + 12);
        } else {
            TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + dist0);
            TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, TOPK_INDEX_DEST_OFFSET + dist1);
            TT_SFPSTORE(
                p_sfpu::LREG7,
                instr_mod_index,
                alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7,
                0,
                TOPK_INDEX_DEST_OFFSET + dist1 + dist0);
        }
    }
}

// Phase 0, step 1 sort building block. Wrapped between two SFPTRANSPs so the swap layer
// operates across the post-transpose lane layout.
template <std::uint32_t SWAP_IMM12>
inline void bitonic_topk_ph0_st1_to_1() {
    TTI_SFPTRANSP;

    // Step 1
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP;
}

// Phase 1, steps 2 then 1. Wrapped between two SFPTRANSPs.
template <std::uint32_t SWAP_IMM12>
inline void bitonic_topk_ph1_st2_to_1() {
    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP;
}

// Phase 2, steps 3, 2, then 1. Step 3 runs on the natural lane layout (no leading TRANSP),
// then a single TRANSP separates it from steps 2/1. The unconditional SFPSWAP(LREG2, LREG3)
// after step 3 is a deliberate post-step-3 reorder copied from the Blackhole reference
// (matches the BH algorithm exactly; do not remove).
template <std::uint32_t SWAP_IMM12>
inline void bitonic_topk_ph2_st3_to_1() {
    // Step 3
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP;
}

// Step-N inner-most layer used by the outer-loop in calculate_bitonic_topk_phases_steps for the
// largest-stride compares. No SFPTRANSP — operates directly on LREG0..LREG3.
// sort_dir==ArgMax: SFPSWAP(LREG0,LREG2) + SFPSWAP(LREG1,LREG3).
// sort_dir==ArgMin: SFPSWAP arg order flipped (LREG2,LREG0) + (LREG3,LREG1) — comparison-direction
// inversion that's distinct from the SFPCONFIG bit-8 EXCHANGE_SRCB_SRCC mechanism phase 4 uses.
// CALLER RESPONSIBILITY: there is no trailing SFPTRANSP, so the caller must follow this with
// either a different-LREG SFPSWAP, an SFPTRANSP, or an explicit TTI_SFPNOP(0,0,0) before any
// SFPSTORE that consumes LREG0..LREG3 (avoids the SFPSWAP→SFPSTORE auto-stall hardware bug).
template <std::uint32_t SWAP_IMM12>
inline void bitonic_topk_step_N(bool sort_dir) {
    // Step N
    if (sort_dir == static_cast<bool>(SortDir::ArgMax)) {
        TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    } else {
        // Min — operand order swapped relative to ArgMax.
        TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}

// Phase 3 step-4-to-1: the building block for the largest-stride bitonic compare layer.
// Performs two passes of "step 4 then step 3" SFPSWAPs followed by an SFPTRANSP. The
// double execution implements the full step-4-to-1 sweep.
//
// sort_dir==ArgMin temporarily flips LaneConfig bit [8] (EXCHANGE_SRCB_SRCC) via SFPCONFIG so
// SFPSWAP compares with reversed min/max polarity; bit [8] is cleared again before return.
// Both 0x104 (set) and 0x004 (clear) keep bit [2] (ENABLE_DEST_INDEX) so the index pairing
// invariant established by init_topk() is preserved across calls (fused: 0x100 / 0x000, tracking off).
template <std::uint32_t SWAP_IMM12, bool FUSED>
inline void bitonic_topk_ph3_st4_to_1(bool sort_dir) {
    // Full-register LaneConfig writes: 0x104 = swap reversal (bit 8) + index tracking (bit 2). The
    // fused mode runs with tracking OFF and must keep it off: 0x100 / 0x000.
    constexpr std::uint32_t lane_config_reversed = FUSED ? 0x100 : 0x104;
    constexpr std::uint32_t lane_config_restored = FUSED ? 0x000 : 0x004;
    if (sort_dir == static_cast<bool>(SortDir::ArgMin)) {
        TTI_SFPCONFIG(lane_config_reversed, 0xF, 1);  // Reverse the max/min behaviour of SWAP
        // SFPCONFIG is a 2-cycle op; per Quasar errata TEN-4581 ("any 2-cycle op
        // followed by SFPSWAP") at least 1 SFPNOP must separate it from the next SFPSWAP.
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }

    // First execution.
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPTRANSP;

    // Second execution.
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(SWAP_IMM12, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPTRANSP;

    if (sort_dir == static_cast<bool>(SortDir::ArgMin)) {
        TTI_SFPCONFIG(lane_config_restored, 0xF, 1);  // Restore the max/min behaviour of SWAP
        // See above — SFPCONFIG → next SFPU op needs an SFPNOP per TEN-4581.
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }
}

/**
 * @brief Run the bitonic local-sort network (phases/steps) over the dest tile on the math thread.
 *
 * @tparam APPROXIMATION_MODE: Approximation flag, kept for SFPU call-shape parity, values = <true/false>
 * @tparam is_fp32_dest_acc_en: Dest register is in 32-bit mode, values = <true/false>
 * @tparam STABLE_SORT: Must be false — stable sort is not supported on the Quasar bitonic path, values = <false>
 * @param initial_sort_dir: Sort direction (ArgMax/ArgMin) the network starts from.
 * @param i_end_phase: Last bitonic phase to run (inclusive).
 * @param i_start_phase: First bitonic phase to run.
 * @param i_end_step: Last step to run within a single requested phase.
 * @param i_start_step: First step to run within a single requested phase.
 * @note Requires LaneConfig bit [2] (ENABLE_DEST_INDEX) set by @ref init_topk beforehand; it persists
 *       on Quasar, so this path does NOT re-assert it via SFPCONFIG. A previous re-assertion clobbered
 *       LaneConfig bit [8] (EXCHANGE_SRCB_SRCC) that @ref bitonic_topk_ph3_st4_to_1 sets for the ArgMin
 *       path, breaking the index swap for multi-tile-row Ascending inputs. Matches the Blackhole reference.
 */
// Top-level local-sort orchestrator. For each (face, col) sub-region of the dest tile,
// walks phases [i_start_phase, i_end_phase]; phases 0..3 each emit 4 groups of
// [load16, swap-body, store16] sequences; phase 4+ falls back to inline step-N..5
// loops followed by the phase-3 helper for steps 4..1.
//
// The BH implementation uses replay slots for load/store and phase bodies. Quasar keeps
// the network inline because replayed SFPSWAP sequences do not preserve the ordering
// this bitonic network requires.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset>
inline void calculate_bitonic_topk_phases_steps(
    const int initial_sort_dir,
    const int i_end_phase,
    const int i_start_phase,
    const int i_end_step,
    const int i_start_step) {
    topk_validate_mode_combo<is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED>();
    constexpr std::uint32_t SWAP_IMM12 = topk_swap_imm12<FUSED, RANK_STAMPED>();

    // Per-call constants (don't depend on face/col/phase) — hoisted out of the (face, col) sweep.
    const bool single_phase = (i_start_phase == i_end_phase);
    const std::uint32_t total_datums_to_compare = 64;

    // Preserve the Blackhole algorithm ordering: fully process every requested
    // phase for one (face, col) sub-region before advancing to the next sub-region.
    set_dst_write_addr(0);
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++) {
        for (int col = 0; col < 2; col++) {
            bool sort_dir = initial_sort_dir;
            for (int phase = i_start_phase; phase < (i_end_phase + 1); phase++) {
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                switch (phase) {
                    case 0: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph0_st1_to_1<SWAP_IMM12>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                        }
                        break;
                    }

                    case 1: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph1_st2_to_1<SWAP_IMM12>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                        }
                        break;
                    }

                    case 2: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph2_st3_to_1<SWAP_IMM12>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                        }
                        break;
                    }

                    case 3: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph3_st4_to_1<SWAP_IMM12, FUSED>(sort_dir);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                            sort_dir = !sort_dir;
                        }
                        break;
                    }

                    default: {
                        // Phases 4..N: steps `num_steps`..5 are emitted inline;
                        // steps 4..1 fall back to the same phase-4 helper as case 3.
                        const std::uint32_t num_steps = phase + 1;
                        const std::uint32_t start_step = single_phase ? i_start_step : num_steps;
                        const std::uint32_t end_step = single_phase ? i_end_step : 4;
                        const std::uint32_t sorted_seq_length = 1 << num_steps;
                        std::uint32_t datums_compared = 0;

                        for (std::uint32_t step = start_step; step > end_step; step--) {
                            // Steps N..5 (inline)
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                            sort_dir = initial_sort_dir;
                            std::uint32_t dist = (step == 5) ? 16 : 32;
                            std::uint32_t inner_iter_count = dist >> 3;
                            datums_compared = 0;
                            std::uint32_t dst_offset = 0;

                            while (datums_compared < total_datums_to_compare) {
                                for (std::uint32_t inner_iter = 0; inner_iter < inner_iter_count; inner_iter++) {
                                    bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 2 * dist);
                                    bitonic_topk_step_N<SWAP_IMM12>(sort_dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false, FUSED, RANK_STAMPED>(4, 2 * dist);

                                    std::uint32_t dst_inc = 8;
                                    dst_offset += dst_inc;
                                    bool dst_cr = false;
                                    if (inner_iter == (inner_iter_count - 1)) {
                                        dst_cr = true;
                                        dst_inc = 4 * dist;
                                        dst_offset = 2 * dist;
                                    } else if (dst_offset == 16) {
                                        dst_cr = true;
                                        dst_inc = 32;
                                    }
                                    bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                    datums_compared += 16;
                                }
                                sort_dir = (datums_compared == sorted_seq_length) ? !sort_dir : sort_dir;
                            }
                        }

                        // Steps 4..1.
                        sort_dir = initial_sort_dir;
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        datums_compared = 0;
                        while (datums_compared < total_datums_to_compare) {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph3_st4_to_1<SWAP_IMM12, FUSED>(sort_dir);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                            datums_compared += 16;
                            sort_dir = (datums_compared == sorted_seq_length) ? !sort_dir : sort_dir;
                        }
                        break;
                    }
                }
            }

            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

/**
 * @brief Run one bitonic merge pass over already-sorted runs in dest on the math thread.
 *
 * @tparam APPROXIMATION_MODE: Approximation flag, kept for SFPU call-shape parity, values = <true/false>
 * @tparam is_fp32_dest_acc_en: Dest register is in 32-bit mode, values = <true/false>
 * @tparam top_min: Sort-direction bit selecting the SFPSWAP operand order (false → ArgMax, true → ArgMin);
 *         corresponds to the Blackhole wrapper's third template parameter, values = <true/false>
 * @tparam STABLE_SORT: Must be false — stable sort is not supported on the Quasar bitonic path, values = <false>
 * @param m_iter: Merge iteration index (selects sorted-run length / compare stride).
 * @param k: TopK width K.
 * @note Requires @ref init_topk beforehand. Does not re-assert SFPCONFIG — see @ref
 * calculate_bitonic_topk_phases_steps.
 */
// Merge: one bitonic merge pass over already-sorted runs in Dest. For each (face, col)
// sub-region, repeatedly load 8 lanes (LREG0,1 values + LREG4,5 indices) via
// bitonic_topk_load8, run an SFPSWAP ALL_ROWS_MAX whose operand order is selected at
// compile time by `top_min` (false → ArgMax: LREG0, LREG1; true → ArgMin: LREG1, LREG0),
// then store back via bitonic_topk_store8.
//
// Merge is short enough to stay inline.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool top_min,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
inline void calculate_bitonic_topk_merge(const int m_iter, const int k) {
    topk_validate_mode_combo<is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED>();
    static_assert(RANK_STAMPED || TAG_BITS == 16, "TAG_BITS applies to the rank-stamped mode only");
    constexpr std::uint32_t SWAP_IMM12 = topk_swap_imm12<FUSED, RANK_STAMPED>();

    if constexpr (RANK_STAMPED) {
        // Program the stamp constants before any load: the SFPCONFIG path clobbers LREG0.
        constexpr std::uint32_t TAG_MASK = topk_tag_mask<TAG_BITS>();
        const std::uint32_t rank_span = 2 * static_cast<std::uint32_t>(k) - 1;
        LLK_ASSERT(rank_span <= TAG_MASK, "merge rank span 2K-1 exceeds the tag field");
        topk_load_lreg_constant(TOPK_LREG_MASK, TAG_MASK);        // tag complement operand
        topk_load_lreg_constant(TOPK_LREG_TAG_CLEAR, ~TAG_MASK);  // tag field clear
        topk_load_lreg_constant(TOPK_LREG_RANK_SPAN, rank_span);  // right-run complement (2K-1)
    }

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++) {
        for (int col = 0; col < 2; col++) {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);

            int k_max = k > 32 ? 32 : k;
            std::uint32_t inner_iter_count = k_max >> 2;  // inner-loop comparisons to sort a length-K sequence
            std::uint32_t total_datums_to_compare = ((64 >> m_iter) < 2 * k_max) ? 2 * k_max : (64 >> m_iter);
            std::uint32_t dist = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter);
            std::uint32_t ld_dist = (dist < 16) ? dist : 2 * dist;  // accounts for face offsets within a tile
            std::uint32_t datums_compared = 0;
            std::uint32_t dst_offset = 0;
            std::uint32_t dst_cr = 0;

            while (datums_compared < total_datums_to_compare) {
                for (std::uint32_t inner_iter = 0; inner_iter < inner_iter_count; inner_iter++) {
                    // The rank-stamped mode re-stamps the words before they become keys.
                    bitonic_topk_load8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED, FUSED>(dst_offset, ld_dist);
                    if constexpr (RANK_STAMPED) {
                        // Re-key both runs' value words with fresh sign-conditioned local ranks so
                        // this merge AND the rebuild after it compare distinct keys whose tie order is
                        // the true index order. The true indices are live in LREG4/5 from the load
                        // above: from here to the swap, ALU-only writes to LREG0..3 (TEN-2932).
                        if (inner_iter == 0) {
                            // Fresh per-pair rank iota: rank = 4 * inner_iter + lane group (each load
                            // covers 4 consecutive run positions per lane group). The left-run base is
                            // 0 for every in-tree caller (one merge call per tile pair).
                            TTI_SFPSHFT(
                                (-4) & 0xFFF /*imm12: shift right 4*/,
                                TOPK_LREG_TILE_ID,
                                p_sfpu::LREG2,
                                TOPK_SFPSHFT_C_BY_IMM);
                        }
                        // largest = !top_min: complement the lanes whose sign matches the kept extreme.
                        // Left run: global direction, lower rank range.
                        TTI_SFPAND(TOPK_LREG_TAG_CLEAR, p_sfpu::LREG0);
                        TTI_SFPOR(p_sfpu::LREG2, p_sfpu::LREG0);
                        topk_complement_tag<!top_min, p_sfpu::LREG0, p_sfpu::LREG0>();
                        // Right run: mirror direction, upper range -- rank' = (2K-1) - rank, a single
                        // XOR because 2K is a power of two and rank < 2K.
                        TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG3, 0 /*instr_mod1: plain copy*/);
                        TTI_SFPXOR(TOPK_LREG_RANK_SPAN, p_sfpu::LREG3);
                        TTI_SFPAND(TOPK_LREG_TAG_CLEAR, p_sfpu::LREG1);
                        TTI_SFPOR(p_sfpu::LREG3, p_sfpu::LREG1);
                        topk_complement_tag<!top_min, p_sfpu::LREG1, p_sfpu::LREG1>();
                        // Advance to the next 4 run positions.
                        TTI_SFPIADD(4 /*imm12*/, p_sfpu::LREG2, p_sfpu::LREG2, TOPK_SFPIADD_IMM_KEEP_CC);
                        topk_flip_keys_0_1<true>();
                    }
                    TTI_SFPSWAP(
                        SWAP_IMM12,
                        top_min ? p_sfpu::LREG1 : p_sfpu::LREG0,
                        top_min ? p_sfpu::LREG0 : p_sfpu::LREG1,
                        p_sfpswap::ALL_ROWS_MAX);
                    bitonic_topk_store8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(dst_offset, ld_dist);

                    datums_compared += 8;
                    if (inner_iter == (inner_iter_count - 1)) {
                        dst_cr += 2 * dist;
                        dst_offset = dst_cr;
                    } else {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

/**
 * @brief Rebuild the bitonic sequence after a merge pass on the math thread.
 *
 * @tparam APPROXIMATION_MODE: Approximation flag, kept for SFPU call-shape parity, values = <true/false>
 * @tparam is_fp32_dest_acc_en: Dest register is in 32-bit mode, values = <true/false>
 * @tparam STABLE_SORT: Must be false — stable sort is not supported on the Quasar bitonic path, values = <false>
 * @param initial_sort_dir: Sort direction (ArgMax/ArgMin). Callers forwarding the test's int
 *        TOPK_SORT_DIRECTION rely on the implicit int→bool conversion at the call site.
 * @param m_iter: Merge iteration index.
 * @param k: TopK width K.
 * @param logk: log2(K); selects the phase via (logk - 1).
 * @param skip_second: When set, halves the datum count (skips the second sub-sequence).
 * @note Requires @ref init_topk beforehand. This path does NOT re-assert SFPCONFIG: doing so clobbered
 *       LaneConfig bit [8] (EXCHANGE_SRCB_SRCC) that @ref bitonic_topk_ph3_st4_to_1 sets for the ArgMin
 *       path — the primary fix for the multi-tile-row Ascending failure. See @ref calculate_bitonic_topk_phases_steps.
 */
// Rebuild: re-runs phase (logK-1) on a merged tile pair to reextract sorted runs.
// A switch on `(logk - 1)` selects one of cases 0/1/2/3/default; each emits a
// load/swap-sequence/store pattern inline for the same replay-safety reason as local sort.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset>
inline void calculate_bitonic_topk_rebuild(
    const bool initial_sort_dir, const int m_iter, const int k, const int logk, const int skip_second) {
    topk_validate_mode_combo<is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED>();
    constexpr std::uint32_t SWAP_IMM12 = topk_swap_imm12<FUSED, RANK_STAMPED>();

    // Per-call constants (don't depend on face/col) — hoisted out of the (face, col) sweep.
    const std::uint32_t total_datums_shift = (skip_second & 0x1);
    const std::uint32_t rebuild_m = m_iter + 1;
    std::uint32_t total_datums_to_compare = ((64 >> rebuild_m) < 2 * k) ? 2 * k : (64 >> rebuild_m);
    total_datums_to_compare >>= total_datums_shift;
    const std::uint32_t dist = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m);
    const std::uint32_t ld_offset = (dist >> 4) * 32 + (dist & 0xF);
    std::uint32_t ld_dist = 0;
    const int phase = logk - 1;

    if (phase == 1 && m_iter < 2) {
        ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
    }

    // Default-branch (phase >= 4) constants — also per-call (phase is fixed per call).
    const std::uint32_t num_steps = phase + 1;
    const std::uint32_t start_step = num_steps;
    const std::uint32_t end_step = 4;
    const std::uint32_t sorted_seq_length = 1 << num_steps;
    const std::uint32_t total_datums_default =
        64;  // shadows total_datums_to_compare in the default branch; intentional, matches BH

    // (face, col) sweep.
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++) {
        for (int col = 0; col < 2; col++) {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
            bool sort_dir = initial_sort_dir;
            std::uint32_t datums_compared = 0;

            switch (phase) {
                case 0: break;

                case 1:
                    if (m_iter >= 2) {
                        while (datums_compared < total_datums_to_compare) {
                            bitonic_topk_load8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(0, ld_offset);
                            bitonic_topk_ph1_st2_to_1<SWAP_IMM12>();
                            bitonic_topk_store8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(0, ld_offset);
                            bitonic_topk_inc_x8_dest(64, false);
                            datums_compared += 16;
                        }
                    } else {
                        while (datums_compared < total_datums_to_compare) {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(ld_offset, ld_dist);
                            bitonic_topk_ph1_st2_to_1<SWAP_IMM12>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(ld_offset, ld_dist);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            datums_compared += 16;
                        }
                    }
                    break;

                case 2:
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, ld_offset);
                        bitonic_topk_ph2_st3_to_1<SWAP_IMM12>();
                        bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, ld_offset);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        datums_compared += 16;
                    }
                    break;

                case 3:
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                        bitonic_topk_ph3_st4_to_1<SWAP_IMM12, FUSED>(sort_dir);
                        bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        datums_compared += 16;
                        sort_dir = !sort_dir;
                    }
                    break;

                default: {
                    // phase >= 4: two-part sort (constants hoisted to the per-call block above).
                    // Part 1: steps `num_steps`..5 emitted inline.
                    // Part 2: steps 4..1.
                    for (std::uint32_t step = start_step; step > end_step; step--) {
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        sort_dir = initial_sort_dir;
                        datums_compared = 0;
                        std::uint32_t dist_inner = (step == 5) ? 16 : 32;
                        std::uint32_t inner_iter_count = dist_inner >> 3;
                        std::uint32_t dst_offset = 0;

                        while (datums_compared < total_datums_default) {
                            for (std::uint32_t inner_iter = 0; inner_iter < inner_iter_count; inner_iter++) {
                                bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 2 * dist_inner);
                                bitonic_topk_step_N<SWAP_IMM12>(sort_dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false, FUSED, RANK_STAMPED>(
                                    4, 2 * dist_inner);

                                std::uint32_t dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (inner_iter == (inner_iter_count - 1)) {
                                    dst_cr = true;
                                    dst_inc = 4 * dist_inner;
                                    dst_offset = 2 * dist_inner;
                                } else if (dst_offset == 16) {
                                    dst_cr = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            sort_dir = (datums_compared == sorted_seq_length) ? !sort_dir : sort_dir;
                        }
                    }

                    // Part 2: steps 4..1.
                    sort_dir = initial_sort_dir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                    while (datums_compared < total_datums_default) {
                        bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                        bitonic_topk_ph3_st4_to_1<SWAP_IMM12, FUSED>(sort_dir);
                        bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                        datums_compared += 16;
                        sort_dir = (datums_compared == sorted_seq_length) ? !sort_dir : sort_dir;
                    }
                    break;
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

// The UInt16-in-32-bit-Dest helpers move a u16 datum into the high half of its Dest word, where the
// Blackhole packer reads a UInt16 from a 32-bit Dest. UInt16 does not exist on Quasar, and a 16-bit
// integer cannot reach a 32-bit Dest there (SrcA rejects it), so they have nothing to move.
// topk_uint16_prepare_value_tile_for_pack is a no-op on Blackhole too unless the kernel is built with
// TOPK_UINT16_FP32_DEST (UInt16 values), which no Quasar kernel can be; the Compute API calls it
// unconditionally, so it stays a no-op here. The other two only make sense on a UInt16 tile, so a
// kernel that calls them fails to build instead of silently packing a wrong half.
inline void topk_uint16_prepare_value_tile_for_pack(std::uint32_t /*tile_index*/) {}

[[gnu::error("UInt16 TopK tiles do not exist on Quasar: the fused-key defuse leaves Int32 indices")]] void
topk_uint16_dest_tile_is_not_supported_on_quasar();

inline void _topk_uint16_move_dest_tile_to_pack_half_(std::uint32_t /*tile_index*/) {
    topk_uint16_dest_tile_is_not_supported_on_quasar();
}

inline void _topk_finalize_hi16_index_tile_(std::uint32_t /*dst_tile_index*/) {
    topk_uint16_dest_tile_is_not_supported_on_quasar();
}

}  // namespace sfpu
}  // namespace ckernel
