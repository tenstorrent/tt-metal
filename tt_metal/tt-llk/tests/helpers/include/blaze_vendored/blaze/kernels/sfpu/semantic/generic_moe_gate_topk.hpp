// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// BUILTIN-BRIDGE LIFT of the generic MoE gate top-k family (lane EX,
// 2026-08-21).  Originals (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk_top8.h
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk_top16.h
//
// Everything the type system can spell is typed; the four mechanisms it
// cannot are bridged one-for-one through the compiler's audited builtins
// (see sfpu_bridge.hpp): indexed swaps (SFPSWAP under ENABLE_DEST_INDEX),
// dual-bank SFPTRANSP, the LaneConfig window toggles, and the partial
// (LO16/HI16) Dst accesses.  Structure, swap mods/directions, transpose
// points, window-toggle points, Dst addresses, and arithmetic order all
// mirror the originals function by function.
//
// Documented intentional differences (state-equivalent, see SEMANTIC-LIFT.md):
//  D1 companion loads: two zero-filled partial loads + OR replace each
//     merging LO16_ONLY/HI16_ONLY load pair (identical register value).
//  D2 normalize drops the SFPCONFIG(0, LREG14, 0) programmable-constant
//     residency trick where the original itself goes typed; the lane-0..7
//     vertical broadcast that SFPCONFIG performs is kept via the typed
//     programmable-constant assignment (vConstFloatPrgm2 = LReg14).
//  D3 normalize's ENABLE_DEST_INDEX window-off toggle moves from after the
//     cross-row adds to before the loads: none of the instructions in
//     between (SFPLOAD/SFPADD/SFPTRANSP) read that LaneConfig bit, and the
//     move removes a TEN-2932 exposure a compiler-allocated body would have.
//  D4 the constexpr-else TTI_NOP in top8_sort_rows is dropped (scheduling
//     filler, no architectural effect).

#include "sfpi.h"
#include "ckernel_sfpu_recip.h"  // sfpu_reciprocal / sfpu_reciprocal_init (same include as the original)
#include "blaze/kernels/sfpu/semantic/sfpu_bridge.hpp"

namespace ckernel {
namespace sfpu {
namespace semantic {

// Dst tile layout — identical constants to the original.
static constexpr uint32_t moe_gate_dst_tile_offset = 64;
static constexpr uint32_t moe_gate_scores_tile = 0;
static constexpr uint32_t moe_gate_indices_tile = 1 * moe_gate_dst_tile_offset;
static constexpr uint32_t moe_gate_bias_tile = 2 * moe_gate_dst_tile_offset;
static constexpr uint32_t moe_gate_interm_tile = 3 * moe_gate_dst_tile_offset;

// The live sort state: what the original pins as LREG0-3 (biased-score sort
// keys) and LREG4-7 (companions packing index LO16 | original-score HI16).
struct SortBank {
    sfpi::vFloat v0, v1, v2, v3;
    sfpi::vUInt c0, c1, c2, c3;
};

// ---- loads/stores (original: _generic_moe_gate_{load,store}_*_even_odd_split_)

template <uint32_t offset>
sfpi_inline void moe_load_16_rows(SortBank &b)
{
    b.v0 = load_value(moe_gate_bias_tile + 0 + offset);
    b.v1 = load_value(moe_gate_bias_tile + 4 + offset);
    b.v2 = load_value(moe_gate_bias_tile + 8 + offset);
    b.v3 = load_value(moe_gate_bias_tile + 12 + offset);
    b.c0 = load_companion(moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
    b.c1 = load_companion(moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
    b.c2 = load_companion(moe_gate_indices_tile + 8 + offset, moe_gate_scores_tile + 8 + offset);
    b.c3 = load_companion(moe_gate_indices_tile + 12 + offset, moe_gate_scores_tile + 12 + offset);
}

template <uint32_t offset>
sfpi_inline void moe_store_16_rows(const SortBank &b)
{
    store_value(b.v0, moe_gate_bias_tile + 0 + offset);
    store_value(b.v1, moe_gate_bias_tile + 4 + offset);
    store_value(b.v2, moe_gate_bias_tile + 8 + offset);
    store_value(b.v3, moe_gate_bias_tile + 12 + offset);
    store_companion(b.c0, moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
    store_companion(b.c1, moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
    store_companion(b.c2, moe_gate_indices_tile + 8 + offset, moe_gate_scores_tile + 8 + offset);
    store_companion(b.c3, moe_gate_indices_tile + 12 + offset, moe_gate_scores_tile + 12 + offset);
}

template <uint32_t offset>
sfpi_inline void moe_load_8_rows(SortBank &b)
{
    b.v0 = load_value(moe_gate_bias_tile + 0 + offset);
    b.v1 = load_value(moe_gate_bias_tile + 4 + offset);
    b.v2 = load_value(moe_gate_bias_tile + 2 + offset);
    b.v3 = load_value(moe_gate_bias_tile + 6 + offset);
    b.c0 = load_companion(moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
    b.c1 = load_companion(moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
    b.c2 = load_companion(moe_gate_indices_tile + 2 + offset, moe_gate_scores_tile + 2 + offset);
    b.c3 = load_companion(moe_gate_indices_tile + 6 + offset, moe_gate_scores_tile + 6 + offset);
}

template <uint32_t offset>
sfpi_inline void moe_store_8_rows(const SortBank &b)
{
    store_value(b.v0, moe_gate_bias_tile + 0 + offset);
    store_value(b.v1, moe_gate_bias_tile + 4 + offset);
    store_companion(b.c0, moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
    store_companion(b.c1, moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
}

// ---- bitonic network primitives (original mods/directions preserved) ----
// p_sfpswap: 0 UNCONDITIONALLY, 1 ALL_ROWS_MAX, 2 ROWS_01_MAX, 3 ROWS_02_MAX.

sfpi_inline void moe_build_bitonic8(SortBank &b)
{
    // P1 - Bitonic 2.
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    // P2 - Bitonic 4.
    indexed_swap<3>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<3>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<3>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<3>(b.v2, b.v3, b.c2, b.c3);
    // P3 - Bitonic 8.
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<2>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<2>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<2>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<2>(b.v2, b.v3, b.c2, b.c3);
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
}

sfpi_inline void moe_bitonic8_steps_3_to_1(SortBank &b)
{
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<2>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<2>(b.v1, b.v3, b.c1, b.c3);
    indexed_swap<2>(b.v0, b.v1, b.c0, b.c1);
    indexed_swap<2>(b.v2, b.v3, b.c2, b.c3);
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
}

// ---- shared tails --------------------------------------------------------

template <int num_selected_experts>
sfpi_inline void moe_zero_tail_lregs(SortBank &b)
{
    static_assert(num_selected_experts >= 1 && num_selected_experts <= 8);
    if constexpr (num_selected_experts > 4) {
        // Original parks the c0 companion halves in the intermediate tile
        // across the transpose window and restores them after.
        store_companion(b.c0, moe_gate_interm_tile, moe_gate_interm_tile + 2);
    } else {
        b.c1 = 0;
    }
    if constexpr (num_selected_experts != 4 && num_selected_experts != 8) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        b.c3 = 0;
        if constexpr (num_selected_experts != 7 && num_selected_experts != 3) {
            b.c2 = 0;
            if constexpr (num_selected_experts != 6 && num_selected_experts != 2) {
                b.c1 = 0;
            }
        }
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
    if constexpr (num_selected_experts > 4) {
        b.c0 = load_companion(moe_gate_interm_tile, moe_gate_interm_tile + 2);
    }
}

template <int num_selected_experts, uint32_t offset>
sfpi_inline void moe_zero_tail(SortBank &b)
{
    moe_load_8_rows<offset>(b);
    moe_zero_tail_lregs<num_selected_experts>(b);
    moe_store_8_rows<offset>(b);
}

template <int num_rows, int scores_offset, bool do_extra_scale = false>
sfpi_inline void moe_normalize(uint32_t eps, uint32_t scale, uint32_t extra_scale = 0)
{
    // D3: leave the ENABLE_DEST_INDEX window before any compiler-allocated
    // arithmetic (the original toggles after its hand-pinned adds).
    set_dest_index_tracking<false>();

    sfpi::vFloat s0 = load_value(scores_offset + 0);
    sfpi::vFloat s1 = load_value(scores_offset + 4);
    sfpi::vFloat s2 = 0.0f, s3 = 0.0f;
    if constexpr (num_rows > 8) {
        s2 = load_value(scores_offset + 8);
        s3 = load_value(scores_offset + 12);
        s0 = s0 + s1;
        s0 = s0 + s2;
        s0 = s0 + s3;
    } else {
        s0 = s0 + s1;
    }

    // Cross-subvector-row reduction, exactly the original's TRANSP + 3 adds.
    sfpi::subvec_transp(s0, s1, s2, s3);
    s0 = s0 + s1;
    s2 = s2 + s3;
    s0 = s0 + s2;

    // The original switches to typed sfpi here; identical math.
    sfpi::vFloat eps_value = Converter::as_float(eps);
    s0 = s0 + eps_value;
    s0 = sfpu_reciprocal<false>(s0);
    sfpi::vFloat scale_value = Converter::as_float(scale);
    if constexpr (do_extra_scale) {
        sfpi::vFloat es = Converter::as_float(extra_scale);
        scale_value = scale_value * es;
    }
    s0 = s0 * scale_value;

    // D2: the original parks s0 in LReg14 via SFPCONFIG(0, LREG14, 0), whose
    // architectural effect is a lanes-0..7 -> all-rows vertical broadcast
    // (SFPCONFIG.md).  The typed programmable-constant assignment is the same
    // instruction; the broadcast is semantic (s0's total lives in the first
    // subvector row after the transpose-reduce).
    sfpi::vConstFloatPrgm2 = s0;
    sfpi::vFloat norm = sfpi::vConstFloatPrgm2;

    sfpi::vFloat r0 = load_value(scores_offset + 0);
    sfpi::vFloat r1 = load_value(scores_offset + 4);
    r0 = r0 * norm;
    r1 = r1 * norm;
    store_value(r0, scores_offset + 0);
    store_value(r1, scores_offset + 4);
    if constexpr (num_rows > 8) {
        sfpi::vFloat r2 = load_value(scores_offset + 8);
        sfpi::vFloat r3 = load_value(scores_offset + 12);
        r2 = r2 * norm;
        r3 = r3 * norm;
        store_value(r2, scores_offset + 8);
        store_value(r3, scores_offset + 12);
    }
}

template <int num_total_experts>
sfpi_inline void moe_generate_indices()
{
    static_assert(num_total_experts >= 128 && num_total_experts <= 1024);
    static_assert(num_total_experts % 128 == 0);
    constexpr uint32_t num_blocks = num_total_experts / 128;

    sfpi::vUInt i0 = sfpi::vUInt(sfpi::vConstTileId);
    sfpi::vUInt i1 = i0 + 1;
    sfpi::vUInt i2 = i0 + 64;
    sfpi::vUInt i3 = i0 + 65;

#pragma GCC unroll 8
    for (uint32_t block = 0; block < num_blocks; block++) {
        const uint32_t offset = block * 8;
        store_uint16(i0, moe_gate_indices_tile + offset + 0);
        store_uint16(i1, moe_gate_indices_tile + offset + 2);
        store_uint16(i2, moe_gate_indices_tile + offset + 4);
        store_uint16(i3, moe_gate_indices_tile + offset + 6);
        i0 = i0 + 128;
        i1 = i1 + 128;
        i2 = i2 + 128;
        i3 = i3 + 128;
    }
}

// ---- top-8 path ----------------------------------------------------------

sfpi_inline void moe_top8_local_sort_16x8_to_8x8(SortBank &b)
{
    moe_build_bitonic8(b);
    // P4 - Partial Bitonic 16 (top8), Step 4.
    indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
}

sfpi_inline void moe_top8_rebuild_and_merge_16x8_to_8x8(SortBank &b)
{
    moe_bitonic8_steps_3_to_1(b);
    indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
}

template <uint32_t offset>
sfpi_inline void moe_top8_load_result_into_upper(SortBank &b)
{
    b.v2 = load_value(moe_gate_bias_tile + 0 + offset);
    b.v3 = load_value(moe_gate_bias_tile + 4 + offset);
    b.c2 = load_companion(moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
    b.c3 = load_companion(moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
}

sfpi_inline void moe_top8_merge_instances(SortBank &b)
{
    // Rotate a copy of the live run one column right, merge; then by 2; then
    // by 4; finally rotate the result lane itself.  SFPSHFT2 must not run
    // under ENABLE_DEST_INDEX (TEN-2932), hence the window toggles — the
    // original's own discipline, bridged 1:1.
    set_dest_index_tracking<false>();
    b.v2 = ror1(b.v0);
    b.v3 = ror1(b.v1);
    b.c2 = ror1(b.c0);
    b.c3 = ror1(b.c1);
    set_dest_index_tracking<true>();
    moe_top8_rebuild_and_merge_16x8_to_8x8(b);

    set_dest_index_tracking<false>();
    b.v2 = ror1(b.v0); ror1_ip(b.v2);
    b.v3 = ror1(b.v1); ror1_ip(b.v3);
    b.c2 = ror1(b.c0); ror1_ip(b.c2);
    b.c3 = ror1(b.c1); ror1_ip(b.c3);
    set_dest_index_tracking<true>();
    moe_top8_rebuild_and_merge_16x8_to_8x8(b);

    set_dest_index_tracking<false>();
    b.v2 = ror1(b.v0); ror1_ip(b.v2); ror1_ip(b.v2); ror1_ip(b.v2);
    b.v3 = ror1(b.v1); ror1_ip(b.v3); ror1_ip(b.v3); ror1_ip(b.v3);
    b.c2 = ror1(b.c0); ror1_ip(b.c2); ror1_ip(b.c2); ror1_ip(b.c2);
    b.c3 = ror1(b.c1); ror1_ip(b.c3); ror1_ip(b.c3); ror1_ip(b.c3);
    set_dest_index_tracking<true>();
    moe_top8_rebuild_and_merge_16x8_to_8x8(b);

    set_dest_index_tracking<false>();
    ror1_ip(b.v0);
    ror1_ip(b.v1);
    ror1_ip(b.c0);
    ror1_ip(b.c1);
    set_dest_index_tracking<true>();
}

template <int num_selected_experts, bool full_sort>
sfpi_inline void moe_top8_sort_rows(SortBank &b)
{
    indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);
    if constexpr (num_selected_experts != 4 || full_sort) {
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        indexed_swap<2>(b.v0, b.v2, b.c0, b.c2);
        indexed_swap<2>(b.v1, b.v3, b.c1, b.c3);
        if constexpr ((num_selected_experts != 6 && num_selected_experts != 2) || full_sort) {
            indexed_swap<2>(b.v0, b.v1, b.c0, b.c1);
            indexed_swap<2>(b.v2, b.v3, b.c2, b.c3);
        }
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    }
    // D4: the original's constexpr-else TTI_NOP is dropped.
}

template <uint32_t load_offset, uint32_t store_offset, bool store_result = true>
sfpi_inline void moe_top8_sort_face(SortBank &b)
{
    moe_load_16_rows<load_offset>(b);
    moe_top8_local_sort_16x8_to_8x8(b);
    moe_store_8_rows<load_offset>(b);

    moe_load_16_rows<load_offset + 2>(b);
    moe_top8_local_sort_16x8_to_8x8(b);

    moe_top8_load_result_into_upper<load_offset>(b);
    moe_top8_rebuild_and_merge_16x8_to_8x8(b);
    if constexpr (store_result) {
        moe_store_8_rows<store_offset>(b);
    }
}

template <uint32_t load_offset, uint32_t store_offset, bool store_result = true>
sfpi_inline void moe_top8_sort_half_face(SortBank &b)
{
    moe_load_8_rows<load_offset>(b);
    moe_top8_local_sort_16x8_to_8x8(b);
    if constexpr (store_result) {
        moe_store_8_rows<store_offset>(b);
    }
}

template <uint32_t face_idx, bool full_face, bool store_result>
sfpi_inline void moe_top8_accumulate_face(SortBank &b)
{
    if constexpr (full_face) {
        moe_top8_sort_face<face_idx * 16, 0, false>(b);
    } else {
        moe_top8_sort_half_face<face_idx * 16, 0, false>(b);
    }
    moe_top8_load_result_into_upper<0>(b);
    moe_top8_rebuild_and_merge_16x8_to_8x8(b);
    if constexpr (store_result) {
        moe_store_8_rows<0>(b);
    }
}

template <int num_total_experts>
sfpi_inline void moe_top8_sort_to_instance(SortBank &b)
{
    static_assert(num_total_experts >= 128 && num_total_experts <= 1024);
    static_assert(num_total_experts % 128 == 0);

    if constexpr (num_total_experts == 128) {
        moe_top8_sort_half_face<0, 0, false>(b);
    } else {
        moe_top8_sort_face<0, 0, (num_total_experts > 256)>(b);
    }
    if constexpr (num_total_experts > 256) {
        moe_top8_accumulate_face<1, (num_total_experts >= 512), (num_total_experts > 512)>(b);
    }
    if constexpr (num_total_experts > 512) {
        moe_top8_accumulate_face<2, (num_total_experts >= 768), (num_total_experts > 768)>(b);
    }
    if constexpr (num_total_experts > 768) {
        moe_top8_accumulate_face<3, (num_total_experts >= 1024), false>(b);
    }
}

template <
    bool normalize,
    int num_selected_experts,
    int num_total_experts,
    bool zero_tail,
    bool full_sort,
    bool do_extra_scale = false>
sfpi_inline void moe_top8(uint32_t eps, uint32_t scale, uint32_t extra_scale, SortBank &b)
{
    moe_top8_sort_to_instance<num_total_experts>(b);
    moe_top8_merge_instances(b);

    if constexpr (num_selected_experts < 8 || full_sort) {
        moe_top8_sort_rows<num_selected_experts, full_sort>(b);
    }

    if constexpr (zero_tail || (normalize && num_selected_experts < 8)) {
        moe_zero_tail_lregs<num_selected_experts>(b);
    }

    moe_store_8_rows<0>(b);

    if constexpr (normalize) {
        moe_normalize<8, moe_gate_scores_tile, do_extra_scale>(eps, scale, extra_scale);
    }

    if constexpr (zero_tail) {
        sfpi::vUInt z = 0;
        store_value(sfpi::vFloat(0.0f), moe_gate_scores_tile + 8);
        store_value(sfpi::vFloat(0.0f), moe_gate_scores_tile + 12);
        store_uint16(z, moe_gate_indices_tile + 8);
        store_uint16(z, moe_gate_indices_tile + 12);
    }
}

// ---- top-16 path ---------------------------------------------------------

template <uint32_t offset>
sfpi_inline void moe_top16_bitonic16_directional(SortBank &b)
{
    if constexpr (offset == 0) {
        indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);  // Step 4.
        indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
        indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);  // Step 3.
        indexed_swap<1>(b.v2, b.v3, b.c2, b.c3);
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);  // Step 2.
        indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
        indexed_swap<1>(b.v0, b.v1, b.c0, b.c1);  // Step 1.
        indexed_swap<1>(b.v2, b.v3, b.c2, b.c3);
    } else if constexpr (offset == 2) {
        // Odd columns sort in the opposite direction (operand order swapped).
        indexed_swap<1>(b.v2, b.v0, b.c2, b.c0);  // Step 4.
        indexed_swap<1>(b.v3, b.v1, b.c3, b.c1);
        indexed_swap<1>(b.v1, b.v0, b.c1, b.c0);  // Step 3.
        indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
        ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
        indexed_swap<1>(b.v2, b.v0, b.c2, b.c0);  // Step 2.
        indexed_swap<1>(b.v3, b.v1, b.c3, b.c1);
        indexed_swap<1>(b.v1, b.v0, b.c1, b.c0);  // Step 1.
        indexed_swap<1>(b.v3, b.v2, b.c3, b.c2);
    }
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
}

sfpi_inline void moe_top16_reverse_sort_order(SortBank &b)
{
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
    indexed_swap<0>(b.v0, b.v3, b.c0, b.c3);  // UNCONDITIONALLY
    indexed_swap<0>(b.v1, b.v2, b.c1, b.c2);
    ckernel::sfpu::semantic::transp8(b.v0, b.v1, b.v2, b.v3, b.c0, b.c1, b.c2, b.c3);
}

template <uint32_t offset>
sfpi_inline void moe_top16_store_16_rows_reverse(const SortBank &b)
{
    store_value(b.v0, moe_gate_bias_tile + 12 + offset);
    store_value(b.v1, moe_gate_bias_tile + 8 + offset);
    store_value(b.v2, moe_gate_bias_tile + 4 + offset);
    store_value(b.v3, moe_gate_bias_tile + 0 + offset);
    store_companion(b.c0, moe_gate_indices_tile + 12 + offset, moe_gate_scores_tile + 12 + offset);
    store_companion(b.c1, moe_gate_indices_tile + 8 + offset, moe_gate_scores_tile + 8 + offset);
    store_companion(b.c2, moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
    store_companion(b.c3, moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
}

sfpi_inline void moe_top16_shift_left_once(SortBank &b)
{
    // Index tracking must be disabled for SFPSHFT2 (original's own comment).
    set_dest_index_tracking<false>();
    ror1_ip(b.v0);
    ror1_ip(b.v1);
    ror1_ip(b.v2);
    ror1_ip(b.v3);
    ror1_ip(b.c0);
    ror1_ip(b.c1);
    ror1_ip(b.c2);
    ror1_ip(b.c3);
    set_dest_index_tracking<true>();
}

template <uint32_t offset>
sfpi_inline void moe_top16_reduce_even_odd_columns_to_instance(SortBank &b)
{
    moe_load_8_rows<offset>(b);
    indexed_swap<1>(b.v0, b.v2, b.c0, b.c2);
    indexed_swap<1>(b.v1, b.v3, b.c1, b.c3);
    moe_store_8_rows<offset>(b);

    moe_load_8_rows<offset + 8>(b);
    // Keep these winners in v2/v3 (companions c2/c3), reload first-half
    // winners into the remaining slots.
    indexed_swap<1>(b.v2, b.v0, b.c2, b.c0);
    indexed_swap<1>(b.v3, b.v1, b.c3, b.c1);

    b.v0 = load_value(moe_gate_bias_tile + 0 + offset);
    b.v1 = load_value(moe_gate_bias_tile + 4 + offset);
    b.c0 = load_companion(moe_gate_indices_tile + 0 + offset, moe_gate_scores_tile + 0 + offset);
    b.c1 = load_companion(moe_gate_indices_tile + 4 + offset, moe_gate_scores_tile + 4 + offset);
    moe_bitonic8_steps_3_to_1(b);
}

template <uint32_t lane_shifts, bool store_result = true>
sfpi_inline void moe_top16_reduce_lanes(SortBank &b)
{
    static_assert(lane_shifts == 1 || lane_shifts == 2 || lane_shifts == 4);

    moe_top16_bitonic16_directional<0>(b);
    moe_store_16_rows<0>(b);

#pragma GCC unroll 4
    for (uint32_t shift = 0; shift < lane_shifts; ++shift) {
        moe_top16_shift_left_once(b);
    }
    moe_top16_reverse_sort_order(b);
    moe_top16_store_16_rows_reverse<2>(b);

    moe_top16_reduce_even_odd_columns_to_instance<0>(b);
    if constexpr (store_result) {
        moe_store_16_rows<0>(b);
    }
}

template <uint32_t load_offset, uint32_t store_offset>
sfpi_inline void moe_top16_sort_face(SortBank &b)
{
    moe_load_16_rows<load_offset>(b);
    moe_build_bitonic8(b);
    moe_top16_bitonic16_directional<0>(b);  // evens descending
    moe_store_16_rows<load_offset>(b);

    moe_load_16_rows<load_offset + 2>(b);
    moe_build_bitonic8(b);
    moe_top16_bitonic16_directional<2>(b);  // odds ascending
    moe_store_16_rows<load_offset + 2>(b);

    moe_top16_reduce_even_odd_columns_to_instance<load_offset>(b);
    moe_store_16_rows<store_offset>(b);
}

template <uint32_t load_offset, uint32_t store_offset>
sfpi_inline void moe_top16_merge_bitonic_face(SortBank &b)
{
    moe_load_16_rows<load_offset>(b);
    moe_top16_bitonic16_directional<0>(b);
    moe_store_16_rows<load_offset>(b);

    moe_load_16_rows<load_offset + 2>(b);
    moe_top16_bitonic16_directional<2>(b);
    moe_store_16_rows<load_offset + 2>(b);

    moe_top16_reduce_even_odd_columns_to_instance<load_offset>(b);
    moe_store_16_rows<store_offset>(b);
}

template <uint32_t load_offset, uint32_t store_offset>
sfpi_inline void moe_top16_sort_half_face(SortBank &b)
{
    moe_load_8_rows<load_offset>(b);
    moe_build_bitonic8(b);
    moe_store_16_rows<store_offset>(b);
}

template <uint32_t face_idx, bool full_face>
sfpi_inline void moe_top16_accumulate_face(SortBank &b)
{
    if constexpr (full_face) {
        moe_top16_sort_face<face_idx * 16, 2>(b);
    } else {
        moe_top16_sort_half_face<face_idx * 16, 2>(b);
    }
    moe_top16_merge_bitonic_face<0, 0>(b);
}

template <int num_total_experts>
sfpi_inline void moe_top16_sort_to_instance(SortBank &b)
{
    static_assert(num_total_experts >= 128 && num_total_experts <= 1024);
    static_assert(num_total_experts % 128 == 0);

    if constexpr (num_total_experts == 128) {
        moe_top16_sort_half_face<0, 0>(b);
    } else {
        moe_top16_sort_face<0, 0>(b);
    }
    if constexpr (num_total_experts > 256) {
        moe_top16_accumulate_face<1, num_total_experts >= 512>(b);
    }
    if constexpr (num_total_experts > 512) {
        moe_top16_accumulate_face<2, num_total_experts >= 768>(b);
    }
    if constexpr (num_total_experts > 768) {
        moe_top16_accumulate_face<3, num_total_experts >= 1024>(b);
    }
}

sfpi_inline void moe_top16_store_outputs(const SortBank &b)
{
    store_companion(b.c0, moe_gate_indices_tile + 0, moe_gate_scores_tile + 0);
    store_companion(b.c1, moe_gate_indices_tile + 4, moe_gate_scores_tile + 4);
    store_companion(b.c2, moe_gate_indices_tile + 8, moe_gate_scores_tile + 8);
    store_companion(b.c3, moe_gate_indices_tile + 12, moe_gate_scores_tile + 12);
}

sfpi_inline void moe_top16_merge_instances(SortBank &b)
{
    moe_top16_reduce_lanes<1>(b);
    moe_top16_reduce_lanes<2>(b);
    moe_top16_reduce_lanes<4, false>(b);

    moe_top16_shift_left_once(b);
    moe_top16_store_outputs(b);
}

template <
    bool normalize,
    int num_selected_experts,
    int num_total_experts,
    bool zero_tail,
    bool full_sort,
    bool do_extra_scale = false>
sfpi_inline void moe_top16(uint32_t eps, uint32_t scale, uint32_t extra_scale, SortBank &b)
{
    static_assert(num_selected_experts >= 9 && num_selected_experts <= 16);

    moe_top16_sort_to_instance<num_total_experts>(b);
    moe_top16_merge_instances(b);

    if constexpr (num_selected_experts < 16 || full_sort) {
        moe_top16_bitonic16_directional<0>(b);
        moe_top16_store_outputs(b);
    }

    if constexpr (num_selected_experts < 16 && (zero_tail || normalize)) {
        moe_zero_tail<num_selected_experts - 8, 8>(b);
    }

    if constexpr (normalize) {
        moe_normalize<16, moe_gate_scores_tile, do_extra_scale>(eps, scale, extra_scale);
    }
}

// ---- entry points --------------------------------------------------------

sfpi_inline void _init_semantic_moe_gate_topk_()
{
    sfpu_reciprocal_init<false>();
}

template <
    bool normalize,
    int num_selected_experts,
    int num_total_experts,
    bool zero_tail,
    bool full_sort,
    bool generate_indices = true,
    bool do_extra_scale = false>
inline void _semantic_moe_gate_topk_(uint32_t eps, uint32_t scale, uint32_t extra_scale = 0)
{
    if constexpr (generate_indices) {
        moe_generate_indices<num_total_experts>();
    }
    set_dest_index_tracking<true>();

    SortBank b{};
    if constexpr (num_selected_experts > 8) {
        moe_top16<normalize, num_selected_experts, num_total_experts, zero_tail, full_sort, do_extra_scale>(
            eps, scale, extra_scale, b);
    } else {
        moe_top8<normalize, num_selected_experts, num_total_experts, zero_tail, full_sort, do_extra_scale>(
            eps, scale, extra_scale, b);
    }
    // Leave the machine with index tracking off (kernel-default LaneConfig).
    // The original ends with whatever state the last phase left; normalize
    // paths end cleared, non-normalize paths end set — callers reinit via
    // SFPU init anyway.  Recorded as a documented trailing-state difference.
    set_dest_index_tracking<false>();
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
