// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// BUILTIN-BRIDGE LIFT of the deepseek MoE gate top-k single-face kernel
// (lane EX, 2026-08-21).  Original (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h
//
// Same recipe as semantic/generic_moe_gate_topk.hpp: typed everywhere the
// type system reaches; the paired value/companion sort choreography bridged
// one-for-one through the compiler's audited builtins (sfpu_bridge.hpp).
// Swap mods/directions, transpose points, window-toggle points, Dst
// addresses and arithmetic order mirror the original function by function.
// Only the !is_fp32_dest_acc_en variants are lifted (the concat-indices
// loaders the entry points use static_assert exactly that).
//
// Documented intentional differences (value/state-equivalent, Fn-numbered):
//  F1 companion loads/stores use the parent bridge's merging load_companion /
//     store_companion (LO16_ONLY/HI16_ONLY) and INT32 round trips — identical
//     words; the two-load merge is pressure-mandatory (parent's D1).
//  F2 _deepseek_moe_gate_sort_top4_groups duplicates the top2 sums with two
//     SFPMOVs (LREG0->LREG2, LREG1->LREG3) INSIDE the ENABLE_DEST_INDEX
//     window.  A compiler-allocated SFPMOV may land in LReg[4..7], which
//     TEN-2932 forbids there; the lift RELOADS the same interm rows instead
//     (SFPLOAD is erratum-exempt).  Value-identical: rows interm+0/+4 are
//     not written between the first loads and the reloads.
//  F3 cross-call LaneConfig contract dissolved: the original sets
//     ENABLE_DEST_INDEX once in _deepseek_moe_gate_sum_top2 and the two
//     later entry points RELY on it staying set across LLK calls.  Each
//     lifted entry point sets the window it needs on entry and leaves it
//     clear on exit (self-contained; same class as the census's cross-call
//     LREG contract finding).
//  F4 window-off toggles move ahead of compiler-allocated arithmetic
//     (parent's D3 discipline): the top2-sum TRANSP/ADD tail and the top8
//     reduce/recip tail run with index tracking already off.  Nothing in the
//     moved-over range reads that LaneConfig bit (loads/adds/transposes are
//     insensitive; the last SFPSWAP is before the move point).
//  F5 the LREG14 residency-and-broadcast (SFPCONFIG(0, LREG14, 0)) is kept
//     via the typed programmable-constant assignment (vConstFloatPrgm2) —
//     the same instruction; its lanes-0..7 vertical broadcast is semantic
//     (parent's D2).
//  F6 register-garbage equivalence: reverse_sort_order's net effect is a
//     per-register subvector-row reversal (transp / swap L0<->L3, L1<->L2 /
//     transp composes to row[j] -> row[3-j] within each register, no
//     cross-register mixing), so the lanes it produces from the original's
//     leftover LREG contents are exactly the lanes the original also
//     discards; the lift's zero-initialized bank members are equivalent for
//     every consumed lane.  Likewise the top8 reduce tail consumes only the
//     first subvector row of the transposed sum (via the LREG14 broadcast);
//     the leftover rows differ but are never read.
//  F7 SFPNOP scheduling filler is not spelled (compiler owns scheduling).
//  F8 the top8 step-4 pair loads are interleaved with their indexed swaps
//     (independent pairs, disjoint Dst rows) — without a transp8 hard-anchor
//     IRA cannot color the 8-live value/companion pairing (compile-proven);
//     see the body comment.
//
// The ONE raw TTI word class: the imm-form LaneConfig toggle
// (set_dest_index_tracking — parent's documented vocabulary gap).
// TTI_SETRWC is spelled through the typed lltt::setrwc wrapper.

#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "lltt.h"
#include "blaze/kernels/sfpu/semantic/sfpu_bridge.hpp"
// Reuse the original's Dst layout constants (read-only include).
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

// The live sort state (original: values LREG0-3, companions LREG4-7).
struct DsBank {
    sfpi::vFloat v0, v1, v2, v3;
    sfpi::vUInt c0, c1, c2, c3;
};

// INT32 (mod0 4) 32-bit opaque companion round trip (original:
// InstrModLoadStore::INT32 loads/stores of the packed idx|score word).
sfpi_inline sfpi::vUInt ds_load_u32(unsigned addr)
{
    return sfpi::vUInt{__builtin_rvtt_sfpload(addr, sfpi::SFPLOAD_MOD0_FMT_INT32,
                                              sfpi::SFPLOAD_ADDR_MODE_NOINC)};
}
sfpi_inline void ds_store_u32(const sfpi::vUInt &c, unsigned addr)
{
    __builtin_rvtt_sfpstore(c.get(), addr, sfpi::SFPSTORE_MOD0_FMT_INT32,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
}
// Raw 32-bit companion access with the value pipeline's mod0 0 (original
// loads bias FLOATS into companion LREGs in sort_top4_groups).
sfpi_inline sfpi::vUInt ds_load_c_raw(unsigned addr)
{
    return sfpi::vUInt{__builtin_rvtt_sfpload(addr, sfpi::SFPLOAD_MOD0_FMT_SRCB,
                                              sfpi::SFPLOAD_ADDR_MODE_NOINC)};
}
sfpi_inline void ds_store_c_raw(const sfpi::vUInt &c, unsigned addr)
{
    __builtin_rvtt_sfpstore(c.get(), addr, sfpi::SFPSTORE_MOD0_FMT_SRCB,
                            sfpi::SFPSTORE_ADDR_MODE_NOINC);
}

// ---- loaders/storers (original bitonic_topk_* helpers, concat variants) ---

template <uint32_t offset>
sfpi_inline void ds_load16_concat(DsBank &b)
{
    b.v0 = load_value(bias_offset + 0 + offset);
    b.v1 = load_value(bias_offset + 4 + offset);
    b.v2 = load_value(bias_offset + 8 + offset);
    b.v3 = load_value(bias_offset + 12 + offset);
    b.c0 = load_companion(indices_offset + 0 + offset, scores_offset + 0 + offset);
    b.c1 = load_companion(indices_offset + 4 + offset, scores_offset + 4 + offset);
    b.c2 = load_companion(indices_offset + 8 + offset, scores_offset + 8 + offset);
    b.c3 = load_companion(indices_offset + 12 + offset, scores_offset + 12 + offset);
}

sfpi_inline void ds_store8_even_concatted(const DsBank &b)
{
    store_value(b.v0, bias_offset + 0);
    store_value(b.v1, bias_offset + 4);
    ds_store_u32(b.c0, interm_offset + 0);
    ds_store_u32(b.c1, interm_offset + 4);
}

sfpi_inline void ds_load8_even_concatted(DsBank &b)
{
    b.v0 = load_value(bias_offset + 0);
    b.v1 = load_value(bias_offset + 4);
    b.c0 = ds_load_u32(interm_offset + 0);
    b.c1 = ds_load_u32(interm_offset + 4);
}

sfpi_inline void ds_store8_even_split(const DsBank &b)
{
    store_value(b.v0, bias_offset + 0);
    store_value(b.v1, bias_offset + 4);
    store_companion(b.c0, indices_offset + 0, scores_offset + 0);
    store_companion(b.c1, indices_offset + 4, scores_offset + 4);
}

// ---- bitonic phases (original mods/directions preserved) -----------------

template <bool start_transpose, bool end_transpose>
sfpi_inline void ds_ph0_st1_to_1(DsBank &b)
{
    if constexpr (start_transpose) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    if constexpr (end_transpose) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool start_transpose, bool end_transpose>
sfpi_inline void ds_ph1_st2_to_1(DsBank &b)
{
    if constexpr (start_transpose) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
    indexed_swap<3>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<3>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<3>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<3>(b.v2, b.v3, b.c2, b.c3);
    if constexpr (end_transpose) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool end_transpose, bool bitonic = true>
sfpi_inline void ds_ph2_st3_to_1(DsBank &b)
{
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);
    if constexpr (bitonic) {
        indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    } else {
        indexed_swap<1>(b.v2, b.v3, b.c2, b.c3);
    }
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    constexpr unsigned swap_mode = bitonic ? 2u /*ROWS_01_MAX*/ : 1u /*ALL_ROWS_MAX*/;
    indexed_swap<swap_mode>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<swap_mode>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<swap_mode>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<swap_mode>(b.v2, b.v3, b.c2, b.c3);
    if constexpr (end_transpose) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool dir, bool end_transpose>
sfpi_inline void ds_top8_ph3_st4_to_1(DsBank &b)
{
    if constexpr (dir == (bool)SortDir::ArgMax) {
        indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);  // Step 4
        indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
        indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);  // Step 3
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);  // Step 2
        indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
        indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);  // Step 1
        indexed_swap<1>(b.v2, b.v3, b.c2, b.c3);
    } else {
        indexed_swap<1>(b.v2, b.v0, b.c2, b.c0);  // Step 4
        indexed_swap<1>(b.v3, b.v1, b.c3, b.c1);
        indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);  // Step 3
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        indexed_swap<1>(b.v2, b.v0, b.c2, b.c0);  // Step 2
        indexed_swap<1>(b.v3, b.v1, b.c3, b.c1);
        indexed_swap<1>(b.v1, b.v0, b.c1, b.c0);  // Step 1
        indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    }
    if constexpr (end_transpose) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool idir>
sfpi_inline void ds_top8_ph0_to_ph3(DsBank &b)
{
    ds_ph0_st1_to_1<true, false>(b);
    ds_ph1_st2_to_1<false, true>(b);
    ds_ph2_st3_to_1<true>(b);
    ds_top8_ph3_st4_to_1<idir, true>(b);
}

sfpi_inline void ds_reverse_sort_order(DsBank &b)
{
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<0>(b.v0, b.v3, b.c0, b.c3);  // UNCONDITIONALLY
    indexed_swap<0>(b.v1, b.v2, b.c1, b.c2);
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
}

// ---- entry points ---------------------------------------------------------

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _semantic_deepseek_moe_gate_sum_top2()
{
    static_assert(!is_fp32_dest_acc_en, "concat-indices path requires 16-bit Dst layout");
    constexpr bool idir = false;  // Sort descending order

    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    set_dest_index_tracking<true>();

    DsBank b{};
    // Phase 0-3 Even Columns
    ds_load16_concat<0>(b);
    ds_top8_ph0_to_ph3<idir>(b);
    ds_store8_even_concatted(b);

    // Phase 0-3 Odd Columns
    ds_load16_concat<2>(b);
    ds_top8_ph0_to_ph3<!idir>(b);

    // Rerun phase 3 on the even/odd top8 pair instead of a full phase 4.
    ds_load8_even_concatted(b);
    ds_top8_ph3_st4_to_1<idir, true>(b);
    ds_store8_even_split(b);

    // Sum top2 (F4: window off before compiler-allocated arithmetic).
    set_dest_index_tracking<false>();
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    b.v0 = b.v0 + b.v1;
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);

    // Replicate the top2 sum down the column (F5: typed prgm-const carries
    // the SFPCONFIG lanes-0..7 vertical broadcast).
    sfpi::vConstFloatPrgm2 = b.v0;
    sfpi::vFloat bc = sfpi::vConstFloatPrgm2;
    // LANE FD EXECUTION FIX (2026-08-21, first CRAQ execution): the stores
    // below MUST run under ENABLE_DEST_INDEX — with the window bit set, a
    // value (bf16) SFPSTORE PRESERVES the Dst word's low 16 bits, and
    // interm's LO16 halves still carry the packed winner indices the later
    // phases re-read.  Lane EX's F3 "each entry self-contained, window left
    // clear" dissolution was wrong: the window bit is store-visible state
    // (craq-sim SFPSTORE bf16 arm honors lane_config bit 2; the original
    // keeps the window on across ALL of sum_top2's stores and exits with it
    // set, inherited by sort_top4/top8).  Re-enable before the stores and
    // exit with the original's window state.
    set_dest_index_tracking<true>();
    store_value(bc, interm_offset);
    store_value(bc, interm_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _semantic_deepseek_moe_gate_sort_top4_groups()
{
    static_assert(!is_fp32_dest_acc_en, "concat-indices path requires 16-bit Dst layout");
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    set_dest_index_tracking<true>();  // F3: original inherits this state

    DsBank b{};
    // Top2 sums + concat indices.
    b.v0 = load_value(interm_offset + 0);
    b.v1 = load_value(interm_offset + 4);
    b.c0 = load_companion(indices_offset + 0, scores_offset + 0);
    b.c1 = load_companion(indices_offset + 4, scores_offset + 4);

    // Top2 sums again (F2: reloads replace the original's in-window SFPMOV
    // copies) and bias scores as raw companion payloads.
    b.v2 = load_value(interm_offset + 0);
    b.v3 = load_value(interm_offset + 4);
    b.c2 = ds_load_c_raw(bias_offset + 0);
    b.c3 = ds_load_c_raw(bias_offset + 4);

    // Sort 8 groups (not bitonic).
    ds_ph0_st1_to_1<true, false>(b);
    ds_ph1_st2_to_1<false, true>(b);
    ds_ph2_st3_to_1<true, false>(b);

    store_companion(b.c0, indices_offset + 0, scores_offset + 0);
    ds_store_c_raw(b.c2, bias_offset + 0);
    // LANE FD EXECUTION FIX: exit with the window ON — the original never
    // clears it here, and the following top8 phase's stores depend on the
    // inherited window state (see the sum_top2 fix note above).
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool do_extra_scale = false>
inline void _semantic_deepseek_moe_gate_top8(uint32_t eps, uint32_t scale, uint32_t extra_scale = 0)
{
    static_assert(!is_fp32_dest_acc_en, "concat-indices path requires 16-bit Dst layout");
    constexpr bool idir = false;  // Sort descending order

    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    set_dest_index_tracking<true>();  // F3

    DsBank b{};
    // Reverse order of sort for top8 values in odd columns.
    b.v2 = load_value(bias_offset + 6);
    b.v3 = load_value(bias_offset + 2);
    b.c2 = load_companion(indices_offset + 6, scores_offset + 6);
    b.c3 = load_companion(indices_offset + 2, scores_offset + 2);
    ds_reverse_sort_order(b);  // F6: v0/v1/c0/c1 lanes discarded below

    b.v0 = load_value(bias_offset + 0);
    b.v1 = load_value(bias_offset + 4);
    b.c0 = load_companion(indices_offset + 0, scores_offset + 0);
    b.c1 = load_companion(indices_offset + 4, scores_offset + 4);
    ds_top8_ph3_st4_to_1<idir, true>(b);

    ds_store8_even_concatted(b);

    // Move and reverse the other column of 8 values; SFPSHFT2 must not run
    // under ENABLE_DEST_INDEX (TEN-2932) — the original's own toggles.
    set_dest_index_tracking<false>();
    b.v3 = ror1(b.v0);
    b.v2 = ror1(b.v1);
    b.c3 = ror1(b.c0);
    b.c2 = ror1(b.c1);
    set_dest_index_tracking<true>();
    ds_reverse_sort_order(b);

    // Step 4 only: top8 without full sorting.  F8: the original loads all
    // four even-column rows and then swaps both pairs; with the reversed odd
    // tops live-through and no transp8 anchor between the loads and the
    // swaps, IRA cannot color the pairing (lreg-pressure-exceeded,
    // compile-proven).  The two swap pairs are independent and their Dst
    // rows disjoint, so the lift interleaves each pair's loads with its swap
    // — same instructions on the same rows, reordered.
    b.v0 = load_value(bias_offset + 0);
    b.c0 = ds_load_u32(interm_offset + 0);
    indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);
    store_companion(b.c0, indices_offset + 0, scores_offset + 0);
    b.v1 = load_value(bias_offset + 4);
    b.c1 = ds_load_u32(interm_offset + 4);
    indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
    store_companion(b.c1, indices_offset + 4, scores_offset + 4);

    // Reduce the top8 values to 1 value (F4: window off before the
    // compiler-allocated tail; original toggles just before the recip).
    set_dest_index_tracking<false>();
    b.v0 = load_value(scores_offset + 0);
    b.v1 = load_value(scores_offset + 4);
    b.v0 = b.v0 + b.v1;
    sfpi::subvec_transp(b.v0, b.v1, b.v2, b.v3);  // companion bank dead here
    b.v0 = b.v0 + b.v1;
    b.v2 = b.v2 + b.v3;
    b.v0 = b.v0 + b.v2;  // row 0 = the true total (F6: rows 1-3 unread)

    // 1 / (sum + eps) * scale — the original's own typed section, verbatim.
    sfpi::vFloat l0 = b.v0;
    sfpi::vFloat eps_value = Converter::as_float(eps);
    l0 = l0 + eps_value;
    l0 = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value = Converter::as_float(scale);
    if constexpr (do_extra_scale) {
        sfpi::vFloat es = Converter::as_float(extra_scale);
        scale_value = scale_value * es;
    }
    l0 = l0 * scale_value;

    // Broadcast to all rows (F5) and multiply the top8 values.
    sfpi::vConstFloatPrgm2 = l0;
    sfpi::vFloat norm = sfpi::vConstFloatPrgm2;
    sfpi::vFloat r0 = load_value(scores_offset + 0);
    sfpi::vFloat r1 = load_value(scores_offset + 4);
    r0 = r0 * norm;
    r1 = r1 * norm;
    store_value(r0, scores_offset + 0);
    store_value(r1, scores_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _init_semantic_deepseek_moe_gate_topk()
{
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
