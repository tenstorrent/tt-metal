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

// NOTE: Stable sort is NOT supported on Quasar. The public entry points keep the
// `STABLE_SORT` template parameter for Blackhole call-shape parity, but assert it
// off (`static_assert(!STABLE_SORT, ...)`); only the unstable bitonic network is
// implemented below.

namespace ckernel {
namespace sfpu {

// Sort direction for topk. Defined locally — Quasar's llk_defs.h does not define SortDir.
enum SortDir : bool {
    ArgMax = false,
    ArgMin = true,
};

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
 * @brief Configure topk-specific SFPU state: ADDR_MOD_6 (@ref init_topk_addr_mod) and the
 *        LaneConfig ENABLE_DEST_INDEX bit (@ref init_topk).
 *
 * @note Run after the base SFPU init and before the topk execute stages.
 */
template <bool APPROXIMATE>
inline void topk_init() {
    init_topk_addr_mod();
    init_topk();
}

// Load 8 lanes (one value LREG pair + one index LREG pair) from Dest at runtime offsets.
// Values land in LREG0,1; indices (offset by dst_indices_offset = 128) land in LREG4,5.
// TopK indices are non-negative Int16 values in the Quasar test harness. Use the
// matching SFPU Int16 memory mode instead of transporting index bits through FP16B.
// Values are Float16_b in this Quasar path, so use explicit FP16B rather than
// sfpmem::DEFAULT to avoid depending on the ALU format left by the index stage.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8(std::uint32_t offset, std::uint32_t dist) {
    constexpr std::uint32_t dst_indices_offset = 128;
    constexpr std::uint32_t instr_mod_value = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices (paired with LREG0,1; shifted by dst_indices_offset).
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset + dist);
}

// Store 8 lanes (the same LREGs that bitonic_topk_load8 fills) back into Dest. Mirrors load8.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8(std::uint32_t offset, std::uint32_t dist) {
    constexpr std::uint32_t dst_indices_offset = 128;
    constexpr std::uint32_t instr_mod_value = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset + dist);
}

// Load 16 lanes (LREG0..3 values + LREG4..7 indices) from Dest at strided offsets
// (0, dist0, dist1, dist1+dist0). The (dist0,dist1)==(4,8) call site is the hot path
// used by phases 5/6 — its addresses are constexpr so we hand them to TTI_SFPLOAD.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16(std::uint32_t dist0, std::uint32_t dist1) {
    constexpr std::uint32_t dst_indices_offset = 128;
    constexpr std::uint32_t instr_mod_value = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

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

    // Indices (paired with LREG0..3; shifted by dst_indices_offset).
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 12);
    } else {
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1 + dist0);
    }
}

// Store 16 lanes (the same LREGs that bitonic_topk_load16 fills) back into Dest.
// When alt_addr_mod=true, the FINAL index store (LREG7) uses ADDR_MOD_6 instead of
// ADDR_MOD_7 — phase 6 configures ADDR_MOD_6 with dest.incr=32 so this auto-advances
// Dest by 32 rows after the last store of a 16-element block.
template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false>
inline void bitonic_topk_store16(std::uint32_t dist0, std::uint32_t dist1) {
    constexpr std::uint32_t dst_indices_offset = 128;
    constexpr std::uint32_t instr_mod_value = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    // Values
    TTI_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 0, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, 0, 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0, dist0);
        TT_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 0, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, 0, dist1 + dist0);
    }

    // Indices — last store optionally swaps to ADDR_MOD_6 for the auto-advance.
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 8);
        TTI_SFPSTORE(
            p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, 0, dst_indices_offset + 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1);
        TT_SFPSTORE(
            p_sfpu::LREG7,
            instr_mod_index,
            alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7,
            0,
            dst_indices_offset + dist1 + dist0);
    }
}

// Phase 0, step 1 sort building block. Wrapped between two SFPTRANSPs so the swap layer
// operates across the post-transpose lane layout.
inline void bitonic_topk_ph0_st1_to_1() {
    TTI_SFPTRANSP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP;
}

// Phase 1, steps 2 then 1. Wrapped between two SFPTRANSPs.
inline void bitonic_topk_ph1_st2_to_1() {
    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP;
}

// Phase 2, steps 3, 2, then 1. Step 3 runs on the natural lane layout (no leading TRANSP),
// then a single TRANSP separates it from steps 2/1. The unconditional SFPSWAP(LREG2, LREG3)
// after step 3 is a deliberate post-step-3 reorder copied from the Blackhole reference
// (matches the BH algorithm exactly; do not remove).
inline void bitonic_topk_ph2_st3_to_1() {
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

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
inline void bitonic_topk_step_N(bool sort_dir) {
    // Step N
    if (sort_dir == static_cast<bool>(SortDir::ArgMax)) {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    } else {
        // Min — operand order swapped relative to ArgMax.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}

// Phase 3 step-4-to-1: the building block for the largest-stride bitonic compare layer.
// Performs two passes of "step 4 then step 3" SFPSWAPs followed by an SFPTRANSP. The
// double execution implements the full step-4-to-1 sweep.
//
// sort_dir==ArgMin temporarily flips LaneConfig bit [8] (EXCHANGE_SRCB_SRCC) via SFPCONFIG so
// SFPSWAP compares with reversed min/max polarity; bit [8] is cleared again before return.
// Both 0x104 (set) and 0x004 (clear) keep bit [2] (ENABLE_DEST_INDEX) so the index pairing
// invariant established by init_topk() is preserved across calls.
inline void bitonic_topk_ph3_st4_to_1(bool sort_dir) {
    if (sort_dir == static_cast<bool>(SortDir::ArgMin)) {
        TTI_SFPCONFIG(0x104, 0xF, 1);  // Reverse the max/min behaviour of SWAP
        // SFPCONFIG is a 2-cycle op; per Quasar errata TEN-4581 ("any 2-cycle op
        // followed by SFPSWAP") at least 1 SFPNOP must separate it from the next SFPSWAP.
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }

    // First execution.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPTRANSP;

    // Second execution.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPTRANSP;

    if (sort_dir == static_cast<bool>(SortDir::ArgMin)) {
        TTI_SFPCONFIG(0x004, 0xF, 1);  // Restore the max/min behaviour of SWAP
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
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_phases_steps(
    const int initial_sort_dir,
    const int i_end_phase,
    const int i_start_phase,
    const int i_end_step,
    const int i_start_step) {
    static_assert(!STABLE_SORT, "Stable TopK is not supported by the Quasar bitonic TopK path");

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
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph0_st1_to_1();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        break;
                    }

                    case 1: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph1_st2_to_1();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        break;
                    }

                    case 2: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph2_st3_to_1();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        break;
                    }

                    case 3: {
                        for (int group = 0; group < 4; group++) {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1(sort_dir);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
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
                                    bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist);
                                    bitonic_topk_step_N(sort_dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist);

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
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1(sort_dir);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
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
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_merge(const int m_iter, const int k) {
    static_assert(!STABLE_SORT, "Stable TopK is not supported by the Quasar bitonic TopK path");

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
                    bitonic_topk_load8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
                    TTI_SFPSWAP(
                        0,
                        top_min ? p_sfpu::LREG1 : p_sfpu::LREG0,
                        top_min ? p_sfpu::LREG0 : p_sfpu::LREG1,
                        p_sfpswap::ALL_ROWS_MAX);
                    bitonic_topk_store8<is_fp32_dest_acc_en>(dst_offset, ld_dist);

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
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_rebuild(
    const bool initial_sort_dir, const int m_iter, const int k, const int logk, const int skip_second) {
    static_assert(!STABLE_SORT, "Stable TopK is not supported by the Quasar bitonic TopK path");

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
                            bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_topk_ph1_st2_to_1();
                            bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_topk_inc_x8_dest(64, false);
                            datums_compared += 16;
                        }
                    } else {
                        while (datums_compared < total_datums_to_compare) {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                            bitonic_topk_ph1_st2_to_1();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
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
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                        bitonic_topk_ph2_st3_to_1();
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        datums_compared += 16;
                    }
                    break;

                case 3:
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                        bitonic_topk_ph3_st4_to_1(sort_dir);
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
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
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist_inner);
                                bitonic_topk_step_N(sort_dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist_inner);

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
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                        bitonic_topk_ph3_st4_to_1(sort_dir);
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
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

}  // namespace sfpu
}  // namespace ckernel
