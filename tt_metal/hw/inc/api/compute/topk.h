// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_topk.h"
#endif

namespace ckernel {

// topK local sort
// clang-format off
/**
 * Performs local sort stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The algorithm used to implement TopK is found here:
 * https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
 *
 * The local sort stage sorts the data into length-K subsequences of
 * alternating directions, in place. If i_start_phase != i_end_phase, all
 * phases in the range i_start_phase to i_end_phase (inclusive) are computed.
 * If i_start_phase == i_end_phase, only that phase is computed, with
 * i_start_step and i_end_step defining how many steps are computed. This can
 * be used to extend the OP support for cases when K > 64.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that local sort is done across columns on 64 values spanning across
 * 2 tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idir            | The sorting direction of the local sort (0 == decreasing, 1 == increasing) | int32    | 0 to 1                                                | True     |
 * | i_end_phase     | The end phase of the local sort (should be set to log(K)-1)                | int32    | 1 to 5                                                | True     |
 * | i_start_phase   | The start phase of the local sort (should be set to 0)                     | int32    | 0 to 5                                                | False    |
 * | i_end_step      | The end step to perform if i_start_phase == i_end_phase                    | int32    | 4 to 6                                                | False    |
 * | i_start_step    | The start step to perform if i_start_phase == i_end_phase                  | int32    | 4 to 6                                                | False    |
 * | stable_sort     | Maintain order of indices for equal values                                 | bool     | true, false                                           | False    |
 * | fused           | Sort packed [bf16 value | u16 index] keys with the unstable network        | bool     | true, false                                           | False    |
 * | rank_stamped    | Sort [bf16 value | rank tag] keys with the unstable network (u32 indices)  | bool     | true, false                                           | False    |
 */
// clang-format on
template <
    bool stable_sort = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    bool fused = false,
    bool rank_stamped = false>
ALWI void topk_local_sort(
    uint32_t idst, int idir, int i_end_phase, int i_start_phase = 0, int i_end_step = 0, int i_start_step = 0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_bitonic_topk_phases_steps,
        (true /* APPROXIMATE */, is_fp32_dest_acc_en, stable_sort, fused, rank_stamped),
        idst,
        VectorMode::RC_custom,
        idir,
        i_end_phase,
        i_start_phase,
        i_end_step,
        i_start_step));
}

// topK merge
// clang-format off
/**
 * Performs merge stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The merge stage combines length-K subsequences that are 2^m_iter apart,
 * such that the first subsequence receives the top K values, and the
 * second subsequence receives the bottom K values.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that merge is done across columns on values spanning across 2
 * tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | m_iter          | The index of the merge & rebuild iteration of the algorithm                | int32    | 0 to 9                                                | True     |
 * | k               | The number of sorted values to return                                      | int32    | {4, 8, 16, 32, 64}                                    | True     |
 * | stable_sort     | Maintain order of indices for equal values                                 | bool     | true, false                                           | False    |
 * | fused           | Sort packed [bf16 value | u16 index] keys with the unstable network        | bool     | true, false                                           | False    |
 * | rank_stamped    | Re-stamp both runs' rank tags and merge with the unstable network          | bool     | true, false                                           | False    |
 * | pre_tagged      | With rank_stamped: tags already ride in the keys (ttnn.sort's per-call     | bool     | true, false                                           | False    |
 * |                 | true-index fuse) — run the rank-stamped transport without re-stamping      |          |                                                       |          |
 */
// clang-format on
template <
    bool idir = false,
    bool stable_sort = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    bool fused = false,
    bool rank_stamped = false,
    bool pre_tagged = false>
ALWI void topk_merge(uint32_t idst, int m_iter, int k) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_bitonic_topk_merge,
        (true /* APPROXIMATE */, is_fp32_dest_acc_en, idir, stable_sort, fused, rank_stamped, pre_tagged),
        idst,
        VectorMode::RC_custom,
        m_iter,
        k));
}

// topK rebuild
// clang-format off
/**
 * Performs rebuild stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The rebuild stage sorts the length-K subsequences that are 2^(m_iter+1)
 * apart, such that they alternate directions.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that rebuild is done across columns on values spanning across 2
 * tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idir            | The sorting direction of the local sort (0 == decreasing, 1 == increasing) | bool     | 0 to 1                                                | True     |
 * | m_iter          | The index of the merge & rebuild iteration of the algorithm                | int32    | 0 to 9                                                | True     |
 * | k               | The number of sorted values to return                                      | int32    | {4, 8, 16, 32, 64}                                    | True     |
 * | logk            | The log of K                                                               | int32    | 2 to 6                                                | True     |
 * | skip_second     | Whether or not to skip second tile                                         | int32    | 0 to 1                                                | True     |
 * | stable_sort     | Maintain order of indices for equal values                                 | bool     | true, false                                           | False    |
 * | fused           | Sort packed [bf16 value | u16 index] keys with the unstable network        | bool     | true, false                                           | False    |
 * | rank_stamped    | Rebuild [bf16 value | rank tag] keys with the unstable network             | bool     | true, false                                           | False    |
 */
// clang-format on
template <
    bool stable_sort = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    bool fused = false,
    bool rank_stamped = false>
ALWI void topk_rebuild(uint32_t idst, bool idir, int m_iter, int k, int logk, int skip_second) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_bitonic_topk_rebuild,
        (true /* APPROXIMATE */, is_fp32_dest_acc_en, stable_sort, fused, rank_stamped),
        idst,
        VectorMode::RC_custom,
        idir,
        m_iter,
        k,
        logk,
        skip_second));
}

/**
 * Please refer to documentation for any_init. fused selects the fused-key init (index tracking
 * off; packed [bf16|u16] keys carry the index inside the sort word). rank_stamped selects the
 * rank-stamped init (index tracking ON — the true u32 indices ride the tracked swaps — plus the
 * rank-tag complement constant).
 */
template <bool fused = false, bool rank_stamped = false>
ALWI void topk_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(topk_local_sort, sfpu::topk_init, (true /* APPROXIMATE */, fused, rank_stamped)));
}

// clang-format off
/**
 * Fuses one 2-tile TopK slab into packed [bf16 value | u16 index'] sort keys, in place in DST.
 * Precondition: value tiles at DST idst..idst+1 as exact-widened [bf16|0x0000] fp32 words (32-bit
 * DEST required) and u16 index tiles at idst+2..idst+3. The index low bits are complemented iff
 * (value_sign == 0) XNOR largest, which makes the sign-magnitude SFPSWAP order torch-stable in the
 * requested GLOBAL direction; run once per freshly loaded slab, never per network call. The index
 * tiles are consumed (dead afterwards). DST must be in acquired state.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | largest         | The requested global sort order (true = largest-first)                     | bool     | true, false                                           | True     |
 * | idst            | The index of the first value tile of the slab in the DST register buffer   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool largest>
ALWI void topk_fuse_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_topk_fuse,
        (true /* APPROXIMATE */, largest),
        idst,
        VectorMode::RC_custom));
}

// clang-format off
/**
 * Splits num_tiles packed [bf16 value | u16 index'] key tiles (starting at DST idst) back into
 * value words ([bf16|0x0000], in place — the following Float32->bf16 pack is then exact) and u16
 * index tiles at idst+2 onward, un-complementing the index with the same largest polarity the fuse
 * used. The index store uses SFPSTORE mode 9 (low->high) so the packer reads UInt16 out of the
 * high half of 32-bit DEST. Run once on the final output tiles; DST must be in acquired state.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | largest         | The requested global sort order (true = largest-first)                     | bool     | true, false                                           | True     |
 * | idst            | The index of the first packed key tile in the DST register buffer          | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | num_tiles       | The number of packed key tiles to split                                    | uint32_t | 1 to 2                                                | True     |
 */
// clang-format on
template <bool largest>
ALWI void topk_defuse_tile(uint32_t idst, uint32_t num_tiles) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_topk_defuse,
        (true /* APPROXIMATE */, largest, 9u /* TOPK_SFPSTORE_MODE_PACK_UINT16 */),
        idst,
        VectorMode::RC_custom,
        num_tiles));
}

// clang-format off
/**
 * Stamps one freshly transposed 2-tile TopK slab's value words with sign-conditioned LOCAL RANK
 * tags in their free low 16 bits, in place in DST: word = [bf16 value | rank XOR (0xFFFF iff
 * value_pos XNOR largest)], rank = the datum's 64-column sequence position. -0.0 is folded into
 * the +0.0 tie class on the way. The plain UNSTABLE network then sorts distinct keys whose
 * equal-value order is the torch-stable index order, while the true (u32) index tiles at DST
 * idst+2..idst+3 ride the index-tracking swaps untouched. Requires 32-bit DEST and the
 * rank-stamped init (topk_tile_init<false, true>). Run once per freshly loaded slab, before
 * every topk_local_sort call in rank-stamped mode; topk_merge re-stamps its runs internally.
 * DST must be in acquired state.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | largest         | The requested global sort order (true = largest-first)                     | bool     | true, false                                           | True     |
 * | idst            | The index of the first value tile of the slab in the DST register buffer   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool largest>
ALWI void topk_stamp_local_positions(std::uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_topk_stamp_local_positions,
        (true /* APPROXIMATE */, largest),
        idst,
        VectorMode::RC_custom));
}

// clang-format off
/**
 * Stamps ONE value tile of a rank-stamped TopK slab with sign-conditioned rank tags covering
 * rank_base + [0, 32), in place in DST (see topk_stamp_local_positions for the tag encoding and
 * preconditions — this is its single-tile form with a caller-chosen base). It exists for the
 * k>32 insertion cascade: each level re-stamps its ACCUMULATOR tile with that tile's round-start
 * chain-position range (32 * level), the fresh incoming chunk is stamped once per round (level 0)
 * with the top range (32 * output_tiles), and the loser tile's tags ride the cascade untouched,
 * so every tag in the round stays globally consistent with the true (value, index) order.
 * rank_base must be a multiple of 32 with rank_base + 31 < 2^16. DST must be in acquired state.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | largest         | The requested global sort order (true = largest-first)                     | bool     | true, false                                           | True     |
 * | idst            | The index of the first value tile of the slab in the DST register buffer   | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_tile_index  | Which slab value tile to stamp (0 or 1)                                    | uint32_t | 0 to 1                                                | True     |
 * | rank_base       | First rank of this tile's range (multiple of 32)                           | uint32_t | 0 to 65504                                            | True     |
 */
// clang-format on
template <bool largest>
ALWI void topk_stamp_tile_rank_range(uint32_t idst, uint32_t dst_tile_index, uint32_t rank_base) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_topk_stamp_tile_rank_range,
        (true /* APPROXIMATE */, largest),
        idst,
        VectorMode::RC_custom,
        dst_tile_index,
        rank_base));
}

/**
 * Clears the low 16 bits (stale rank tags) of one rank-stamped value tile in DST, leaving exact
 * [bf16|0x0000] words so the following Float32->bf16 pack cannot RNE-round on tag bits. Must run
 * on MATH while DEST is still acquired, after the final transpose back to row layout (same
 * calling convention as topk_uint16_move_dest_tile_to_pack_half).
 */
ALWI void topk_strip_rank_tags(std::uint32_t idst) { MATH((ckernel::sfpu::_topk_strip_rank_tags_(idst))); }

// clang-format off
/**
 * Sets the tie-break polarity used by the stable TopK comparator. Required once per compute
 * kernel, after topk_tile_init, when any TopK call uses stable_sort=true. The polarity encodes
 * the requested GLOBAL sort order and must not follow the per-call sort direction (idir), which
 * may alternate to build bitonic sequences.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | descending      | The requested global sort order (true = largest-first / descending)        | bool     | true, false                                           | True     |
 */
// clang-format on
ALWI void topk_set_stable_descending_mode(bool descending) {
    MATH((ckernel::sfpu::set_topk_stable_descending_mode(descending)));
}

/**
 * UInt16 values in 32-bit DEST: move cleaned values into the packer-visible high half (SFPSTORE mode 9).
 * No-op unless the compute kernel was built with TOPK_UINT16_FP32_DEST. Must run on MATH while DEST is
 * still acquired (before tile_regs_commit / pack_tile). See #50215.
 */
ALWI void topk_uint16_prepare_value_tile_for_pack(uint32_t idst) {
    MATH((ckernel::sfpu::topk_uint16_prepare_value_tile_for_pack(idst)));
}

/**
 * Moves a u16 DEST tile's datums from the low half of each 32-bit DEST word (where a u16 transpose
 * lands them) into the packer-visible high half, stripping the stale garbage above them. Used by
 * the fused-key TopK final extraction; unconditional (not gated on TOPK_UINT16_FP32_DEST). Must
 * run on MATH while DEST is still acquired (before tile_regs_commit / pack_tile).
 */
ALWI void topk_uint16_move_dest_tile_to_pack_half(uint32_t idst) {
    MATH((ckernel::sfpu::_topk_uint16_move_dest_tile_to_pack_half_(idst)));
}

}  // namespace ckernel
