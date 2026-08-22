// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CROSS-LANE MIGRATION of the deepseek MoE gate top-k single-face bridge
// lift (lane FK, 2026-08-22).  Bridge-lift parent (byte-untouched):
//   semantic/deepseek_moe_gate_topk_single_face.hpp (lane EX + lane FD
//   errata, incl. the store-visible ENABLE_DEST_INDEX window discipline).
// Original (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h
//
// Same surface mapping as generic_moe_gate_topk_crosslane.hpp (see that
// header for the full table and the mod-0 vocabulary-gap note):
//   indexed_swap<1> -> sfpi::sort2_kv<SortOrder::Descending> (same operand
//   order); indexed_swap<3> -> sfpi::sort2_kv_rows<RowPattern::Min02Max13>
//   (operands swapped); indexed_swap<2> -> RowPattern::Min01Max23 (swapped);
//   indexed_swap<0> -> direct audited builtin (no surface op);
//   transp8 -> sfpi::transp8; window toggles -> sfpi::set_dest_index_window
//   (typed imm-form markers X4 scopes windows by); companion Dst access ->
//   sfpi::dst_load_packed/dst_store_packed; value Dst access ->
//   sfpi::dst_reg[addr/2]; ror1 -> sfpi::subvec_rotr<1>.
// The INT32/raw-mod0 32-bit companion round trips keep their direct
// builtins (no surface vocabulary; they are not bridge helpers).
//
// F1-F8 (vs the hand original) plus the lane-FD window-state fixes are
// inherited from the bridge lift verbatim; sort_top4_groups carries the
// bridge lift's KNOWN execution refutation (lane FD: hybrid-phase H=2
// fails) -- the vehicle races only the proven phases.  M3 (transp8 with
// zeroed companions replacing subvec_transp in the top8 reduce tail) as in
// the generic migration.

#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "lltt.h"
// Reuse the original's Dst layout constants (read-only include).
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h"

namespace ckernel {
namespace sfpu {
namespace semantic {
namespace crosslane {

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

// Typed Dst value access at hand-kernel addresses.
sfpi_inline sfpi::vFloat ds_load(unsigned addr)
{
    return sfpi::dst_reg[addr / 2];
}
sfpi_inline void ds_store(const sfpi::vFloat &v, unsigned addr)
{
    sfpi::dst_reg[addr / 2] = v;
}

// Unconditional dual-bank swap (SFPSWAP Mod1=0; surface vocabulary gap --
// see generic_moe_gate_topk_crosslane.hpp header).
sfpi_inline void ds_swap_unconditional(sfpi::vFloat &va, sfpi::vFloat &vb,
                                       sfpi::vUInt &ca, sfpi::vUInt &cb)
{
    auto r = __builtin_rvtt_sfpswap_indexed(vb.get(), va.get(),
                                            cb.get(), ca.get(),
                                            sfpi::SFPSWAP_MOD1_SWAP);
    vb = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 0));
    va = sfpi::vFloat(__builtin_rvtt_sfpselect4(r, 1));
    cb = sfpi::vUInt(__builtin_rvtt_sfpselect4(r, 2));
    ca = sfpi::vUInt(__builtin_rvtt_sfpselect4(r, 3));
}

// ---- loaders/storers (original bitonic_topk_* helpers, concat variants) ---

template <uint32_t offset>
sfpi_inline void ds_load16_concat(DsBank &b)
{
    b.v0 = ds_load(bias_offset + 0 + offset);
    b.v1 = ds_load(bias_offset + 4 + offset);
    b.v2 = ds_load(bias_offset + 8 + offset);
    b.v3 = ds_load(bias_offset + 12 + offset);
    b.c0 = sfpi::dst_load_packed(indices_offset + 0 + offset, scores_offset + 0 + offset);
    b.c1 = sfpi::dst_load_packed(indices_offset + 4 + offset, scores_offset + 4 + offset);
    b.c2 = sfpi::dst_load_packed(indices_offset + 8 + offset, scores_offset + 8 + offset);
    b.c3 = sfpi::dst_load_packed(indices_offset + 12 + offset, scores_offset + 12 + offset);
}

sfpi_inline void ds_store8_even_concatted(const DsBank &b)
{
    ds_store(b.v0, bias_offset + 0);
    ds_store(b.v1, bias_offset + 4);
    ds_store_u32(b.c0, interm_offset + 0);
    ds_store_u32(b.c1, interm_offset + 4);
}

sfpi_inline void ds_load8_even_concatted(DsBank &b)
{
    b.v0 = ds_load(bias_offset + 0);
    b.v1 = ds_load(bias_offset + 4);
    b.c0 = ds_load_u32(interm_offset + 0);
    b.c1 = ds_load_u32(interm_offset + 4);
}

sfpi_inline void ds_store8_even_split(const DsBank &b)
{
    ds_store(b.v0, bias_offset + 0);
    ds_store(b.v1, bias_offset + 4);
    sfpi::dst_store_packed(b.c0, indices_offset + 0, scores_offset + 0);
    sfpi::dst_store_packed(b.c1, indices_offset + 4, scores_offset + 4);
}

// ---- bitonic phases (original mods/directions preserved) -----------------

template <bool start_transpose, bool end_transpose>
sfpi_inline void ds_ph0_st1_to_1(DsBank &b)
{
    using sfpi::sort2_kv;
    using sfpi::SortOrder;
    if constexpr (start_transpose) {
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
    sort2_kv<SortOrder::Descending>(b.v0, b.v1, b.c0, b.c1);
    sort2_kv<SortOrder::Descending>(b.v3, b.v2, b.c3, b.c2);
    if constexpr (end_transpose) {
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool start_transpose, bool end_transpose>
sfpi_inline void ds_ph1_st2_to_1(DsBank &b)
{
    using sfpi::sort2_kv_rows;
    using sfpi::RowPattern;
    if constexpr (start_transpose) {
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
    sort2_kv_rows<RowPattern::Min02Max13>(b.v2, b.v0, b.c2, b.c0);
    sort2_kv_rows<RowPattern::Min02Max13>(b.v3, b.v1, b.c3, b.c1);
    sort2_kv_rows<RowPattern::Min02Max13>(b.v1, b.v0, b.c1, b.c0);
    sort2_kv_rows<RowPattern::Min02Max13>(b.v3, b.v2, b.c3, b.c2);
    if constexpr (end_transpose) {
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool end_transpose, bool bitonic = true>
sfpi_inline void ds_ph2_st3_to_1(DsBank &b)
{
    using sfpi::sort2_kv;
    using sfpi::sort2_kv_rows;
    using sfpi::SortOrder;
    using sfpi::RowPattern;
    sort2_kv<SortOrder::Descending>(b.v0, b.v1, b.c0, b.c1);
    if constexpr (bitonic) {
        sort2_kv<SortOrder::Descending>(b.v3, b.v2, b.c3, b.c2);
    } else {
        sort2_kv<SortOrder::Descending>(b.v2, b.v3, b.c2, b.c3);
    }
    sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    if constexpr (bitonic) {
        sort2_kv_rows<RowPattern::Min01Max23>(b.v2, b.v0, b.c2, b.c0);
        sort2_kv_rows<RowPattern::Min01Max23>(b.v3, b.v1, b.c3, b.c1);
        sort2_kv_rows<RowPattern::Min01Max23>(b.v1, b.v0, b.c1, b.c0);
        sort2_kv_rows<RowPattern::Min01Max23>(b.v3, b.v2, b.c3, b.c2);
    } else {
        sort2_kv<SortOrder::Descending>(b.v0, b.v2, b.c0, b.c2);
        sort2_kv<SortOrder::Descending>(b.v1, b.v3, b.c1, b.c3);
        sort2_kv<SortOrder::Descending>(b.v0, b.v1, b.c0, b.c1);
        sort2_kv<SortOrder::Descending>(b.v2, b.v3, b.c2, b.c3);
    }
    if constexpr (end_transpose) {
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
}

template <bool dir, bool end_transpose>
sfpi_inline void ds_top8_ph3_st4_to_1(DsBank &b)
{
    using sfpi::sort2_kv;
    using sfpi::SortOrder;
    if constexpr (dir == (bool)SortDir::ArgMax) {
        sort2_kv<SortOrder::Descending>(b.v0, b.v2, b.c0, b.c2);  // Step 4
        sort2_kv<SortOrder::Descending>(b.v1, b.v3, b.c1, b.c3);
        sort2_kv<SortOrder::Descending>(b.v0, b.v1, b.c0, b.c1);  // Step 3
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        sort2_kv<SortOrder::Descending>(b.v0, b.v2, b.c0, b.c2);  // Step 2
        sort2_kv<SortOrder::Descending>(b.v1, b.v3, b.c1, b.c3);
        sort2_kv<SortOrder::Descending>(b.v0, b.v1, b.c0, b.c1);  // Step 1
        sort2_kv<SortOrder::Descending>(b.v2, b.v3, b.c2, b.c3);
    } else {
        sort2_kv<SortOrder::Descending>(b.v2, b.v0, b.c2, b.c0);  // Step 4
        sort2_kv<SortOrder::Descending>(b.v3, b.v1, b.c3, b.c1);
        sort2_kv<SortOrder::Descending>(b.v3, b.v2, b.c3, b.c2);  // Step 3
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        sort2_kv<SortOrder::Descending>(b.v2, b.v0, b.c2, b.c0);  // Step 2
        sort2_kv<SortOrder::Descending>(b.v3, b.v1, b.c3, b.c1);
        sort2_kv<SortOrder::Descending>(b.v1, b.v0, b.c1, b.c0);  // Step 1
        sort2_kv<SortOrder::Descending>(b.v3, b.v2, b.c3, b.c2);
    }
    if constexpr (end_transpose) {
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
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
    sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    ds_swap_unconditional(b.v0, b.v3, b.c0, b.c3);  // UNCONDITIONALLY
    ds_swap_unconditional(b.v1, b.v2, b.c1, b.c2);
    sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
}

// ---- entry points ---------------------------------------------------------

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _semantic_deepseek_moe_gate_sum_top2()
{
    static_assert(!is_fp32_dest_acc_en, "concat-indices path requires 16-bit Dst layout");
    constexpr bool idir = false;  // Sort descending order

    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    sfpi::set_dest_index_window<true>();

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
    sfpi::set_dest_index_window<false>();
    sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    b.v0 = b.v0 + b.v1;
    sfpi::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);

    // Replicate the top2 sum down the column (F5: typed prgm-const carries
    // the SFPCONFIG lanes-0..7 vertical broadcast).
    sfpi::vConstFloatPrgm2 = b.v0;
    sfpi::vFloat bc = sfpi::vConstFloatPrgm2;
    // LANE FD EXECUTION FIX (inherited): the stores below MUST run under
    // ENABLE_DEST_INDEX — with the window bit set, a value (bf16) SFPSTORE
    // PRESERVES the Dst word's low 16 bits, and interm's LO16 halves still
    // carry the packed winner indices the later phases re-read.  Re-enable
    // before the stores and exit with the original's window state.
    sfpi::set_dest_index_window<true>();
    ds_store(bc, interm_offset);
    ds_store(bc, interm_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _semantic_deepseek_moe_gate_sort_top4_groups()
{
    static_assert(!is_fp32_dest_acc_en, "concat-indices path requires 16-bit Dst layout");
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    sfpi::set_dest_index_window<true>();  // F3: original inherits this state

    DsBank b{};
    // Top2 sums + concat indices.
    b.v0 = ds_load(interm_offset + 0);
    b.v1 = ds_load(interm_offset + 4);
    b.c0 = sfpi::dst_load_packed(indices_offset + 0, scores_offset + 0);
    b.c1 = sfpi::dst_load_packed(indices_offset + 4, scores_offset + 4);

    // Top2 sums again (F2: reloads replace the original's in-window SFPMOV
    // copies) and bias scores as raw companion payloads.
    b.v2 = ds_load(interm_offset + 0);
    b.v3 = ds_load(interm_offset + 4);
    b.c2 = ds_load_c_raw(bias_offset + 0);
    b.c3 = ds_load_c_raw(bias_offset + 4);

    // Sort 8 groups (not bitonic).
    ds_ph0_st1_to_1<true, false>(b);
    ds_ph1_st2_to_1<false, true>(b);
    ds_ph2_st3_to_1<true, false>(b);

    sfpi::dst_store_packed(b.c0, indices_offset + 0, scores_offset + 0);
    ds_store_c_raw(b.c2, bias_offset + 0);
    // LANE FD EXECUTION FIX (inherited): exit with the window ON — the
    // original never clears it here, and the following top8 phase's stores
    // depend on the inherited window state.
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool do_extra_scale = false>
inline void _semantic_deepseek_moe_gate_top8(uint32_t eps, uint32_t scale, uint32_t extra_scale = 0)
{
    static_assert(!is_fp32_dest_acc_en, "concat-indices path requires 16-bit Dst layout");
    constexpr bool idir = false;  // Sort descending order

    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    sfpi::set_dest_index_window<true>();  // F3

    DsBank b{};
    // Reverse order of sort for top8 values in odd columns.
    b.v2 = ds_load(bias_offset + 6);
    b.v3 = ds_load(bias_offset + 2);
    b.c2 = sfpi::dst_load_packed(indices_offset + 6, scores_offset + 6);
    b.c3 = sfpi::dst_load_packed(indices_offset + 2, scores_offset + 2);
    ds_reverse_sort_order(b);  // F6: v0/v1/c0/c1 lanes discarded below

    b.v0 = ds_load(bias_offset + 0);
    b.v1 = ds_load(bias_offset + 4);
    b.c0 = sfpi::dst_load_packed(indices_offset + 0, scores_offset + 0);
    b.c1 = sfpi::dst_load_packed(indices_offset + 4, scores_offset + 4);
    ds_top8_ph3_st4_to_1<idir, true>(b);

    ds_store8_even_concatted(b);

    // Move and reverse the other column of 8 values; SFPSHFT2 must not run
    // under ENABLE_DEST_INDEX (TEN-2932) — the original's own toggles.
    sfpi::set_dest_index_window<false>();
    b.v3 = sfpi::subvec_rotr<1>(b.v0);
    b.v2 = sfpi::subvec_rotr<1>(b.v1);
    b.c3 = sfpi::subvec_rotr<1>(b.c0);
    b.c2 = sfpi::subvec_rotr<1>(b.c1);
    sfpi::set_dest_index_window<true>();
    ds_reverse_sort_order(b);

    // Step 4 only: top8 without full sorting.  F8 (inherited): interleave
    // each pair's loads with its swap — without a transp8 hard-anchor IRA
    // cannot color the 8-live value/companion pairing.
    b.v0 = ds_load(bias_offset + 0);
    b.c0 = ds_load_u32(interm_offset + 0);
    sfpi::sort2_kv<sfpi::SortOrder::Descending>(b.v0, b.v2, b.c0, b.c2);
    sfpi::dst_store_packed(b.c0, indices_offset + 0, scores_offset + 0);
    b.v1 = ds_load(bias_offset + 4);
    b.c1 = ds_load_u32(interm_offset + 4);
    sfpi::sort2_kv<sfpi::SortOrder::Descending>(b.v1, b.v3, b.c1, b.c3);
    sfpi::dst_store_packed(b.c1, indices_offset + 4, scores_offset + 4);

    // Reduce the top8 values to 1 value (F4: window off before the
    // compiler-allocated tail; original toggles just before the recip).
    sfpi::set_dest_index_window<false>();
    b.v0 = ds_load(scores_offset + 0);
    b.v1 = ds_load(scores_offset + 4);
    b.v0 = b.v0 + b.v1;
    // M3: audited dual-bank transp8 (companion bank dead here — zeroed).
    {
        sfpi::vUInt z0 = 0, z1 = 0, z2 = 0, z3 = 0;
        sfpi::transp8(b.v0, b.v1, b.v2, b.v3, z0, z1, z2, z3);
    }
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
    sfpi::vFloat r0 = ds_load(scores_offset + 0);
    sfpi::vFloat r1 = ds_load(scores_offset + 4);
    r0 = r0 * norm;
    r1 = r1 * norm;
    ds_store(r0, scores_offset + 0);
    ds_store(r1, scores_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _init_semantic_deepseek_moe_gate_topk()
{
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace crosslane
}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
