// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#include "sanitizer/api.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_pack_indices.h"

namespace ckernel {

template <std::uint32_t K, bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_topk_xl_init_<K, fused>);
}

template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_local_sort(
    std::uint32_t dst_index, bool ascending, VectorMode vector_mode = VectorMode::RC_custom) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_local_sort_<K>, dst_index, vector_mode, dst_index, ascending);
}

// Per-column-isolated variant: early_exit_K64 sorts each 64-row column
// independently and skips the cross-column merge (sparse-K reader).
template <std::uint32_t K, bool early_exit_K64>
inline void llk_math_eltwise_unary_sfpu_topk_xl_local_sort_generic(
    std::uint32_t dst_index, bool ascending, VectorMode vector_mode = VectorMode::RC_custom) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_local_sort_generic_<K, early_exit_K64>, dst_index, vector_mode, dst_index, ascending);
}

template <std::uint32_t K, bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_merge(
    std::uint32_t dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_merge_<K, fused>, dst_index, vector_mode, dst_index);
}

template <std::uint32_t K, bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_rebuild(
    std::uint32_t dst_index, bool ascending, VectorMode vector_mode = VectorMode::RC_custom) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_rebuild_<K, fused>, dst_index, vector_mode, dst_index, ascending);
}

// Sort each 64 row column of a fused K = 1024 tile that is already bitonic. Call it right after the fused merge: it
// relies on that merge's SFPLOADMACRO setup and programs none.
template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_rebuild_columns(
    std::uint32_t dst_index, bool ascending, VectorMode vector_mode = VectorMode::RC_custom) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_rebuild_columns_<K>, dst_index, vector_mode, dst_index, ascending);
}

inline void llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_topk_xl_add_lsb_indices_init_);
}

template <std::uint32_t K, std::uint32_t core_id, bool row_major = false>
inline void llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices(std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_add_lsb_indices_<K, core_id, row_major>, dst_index, VectorMode::RC_custom);
}

/**
 * @brief Initialize the shared SFPU state for supplied-index packing.
 *
 * @note Call after loading the inputs and before @ref llk_math_topk_xl_pack_supplied_indices.
 *       Programs ADDR_MOD_0; reinitialize datacopy before the next copy.
 *       Restore TopK state afterward as described by @ref topk_xl_pack_supplied_indices.
 */
inline void llk_math_topk_xl_pack_supplied_indices_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_topk_xl_pack_supplied_indices_init_);
}

/**
 * @brief Pack score and raw-index regions into fused TopK entries in FP32 DEST.
 *
 * @tparam K: Number of entries, 512, 1024, or 2048.
 * @tparam is_fp32_dest_acc_en: Actual DEST accumulation mode; must be true, including runtime FP32 switches.
 * @tparam Dst: Acquired DEST scope, SyncHalf or SyncFull; does not acquire or reconfigure DEST.
 * @param scores: First score tile in DEST.
 * @param indices: First raw UInt32 ID tile; IDs start at element zero for either score half.
 * @param output: First output tile; may alias either input at the same start only.
 * @param score_half: Score half for K=512, 0 or 1; zero for larger K. Does not offset IDs.
 * @note Acquire the entire referenced DEST range and call @ref llk_math_topk_xl_pack_supplied_indices_init
 *       before this function. Restore TopK state afterward as described by @ref topk_xl_pack_supplied_indices.
 */
template <std::uint32_t K, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DstSync Dst = DST_SYNC_MODE>
inline void llk_math_topk_xl_pack_supplied_indices(
    std::uint32_t scores, std::uint32_t indices, std::uint32_t output, std::uint32_t score_half) {
    SAN_HOOK(unsupported());
    static_assert(is_fp32_dest_acc_en, "supplied-index packing requires FP32 DEST accumulation");
    static_assert(Dst == DstSync::SyncHalf || Dst == DstSync::SyncFull, "invalid DEST synchronization scope");
    LLK_ASSERT(llk_assert_fp32_dest_acc_en, "supplied-index packing requires runtime FP32 DEST accumulation");
    constexpr std::uint32_t tiles = K == 2048 ? 2 : 1;
    // Check the starts as well as the ends so adding the second tile cannot wrap an invalid index.
    _llk_math_eltwise_sfpu_assert_dst_index_<Dst>(scores, "score start out of range");
    _llk_math_eltwise_sfpu_assert_dst_index_<Dst>(indices, "index start out of range");
    _llk_math_eltwise_sfpu_assert_dst_index_<Dst>(output, "output start out of range");
    _llk_math_eltwise_sfpu_assert_dst_index_<Dst>(scores + tiles - 1, "score end out of range");
    _llk_math_eltwise_sfpu_assert_dst_index_<Dst>(indices + tiles - 1, "index end out of range");
    _llk_math_eltwise_sfpu_assert_dst_index_<Dst>(output + tiles - 1, "output end out of range");
    LLK_ASSERT(
        output == scores || (output > scores ? output - scores : scores - output) >= tiles,
        "output partially overlaps scores");
    LLK_ASSERT(
        output == indices || (output > indices ? output - indices : indices - output) >= tiles,
        "output partially overlaps indices");
    LLK_ASSERT(score_half == 0 || (K == 512 && score_half == 1), "invalid score half");
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::_topk_xl_pack_supplied_indices_<K>, scores, indices, output, VectorMode::RC_custom, score_half);
}

// Reprogram only the MOP Expander after a topk_xl copy, instead of a full
// topk_xl_init. See ckernel_sfpu_topk_xl.h for what copy init clobbers. The unfused form also reprograms the
// SFPLOADMACRO Sequence and Misc registers and clobbers LREG0.
template <bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_reinit_mop_after_copy() {
    SAN_HOOK(unsupported());
    ckernel::sfpu::topk_mop_config<fused>();
}

// Restore the ADDR_MODs and MOP state the unfused rebuild needs after a copy, and the unfused SFPLOADMACRO Sequence
// and Misc registers (clobbering LREG0).
inline void llk_math_eltwise_unary_sfpu_topk_xl_reinit_unfused_rebuild_after_copy() {
    SAN_HOOK(unsupported());
    ckernel::sfpu::topk_reinit_unfused_rebuild_after_copy();
}

// Runtime-chunk-id stamp (fused end-to-end rows: one instantiation, id per chunk).
template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices_rt(std::uint32_t dst_index, std::uint32_t chunk_id) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_add_lsb_indices_rt_<K>, dst_index, VectorMode::RC_custom, chunk_id);
}

inline void llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values_init() {
    SAN_HOOK(unsupported());
    ckernel::sfpu::_topk_xl_remove_msb_values_init_();
}

template <std::uint32_t K, DstSync Dst>
inline void llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values(std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH | p_stall::PACK);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::_topk_xl_remove_msb_values_<K, Dst>();
}

inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_init(std::uint32_t group_id_bit_shift) {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_init_, group_id_bit_shift);
}

// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split API.
// These wrappers expose the row-major SFPU helpers added to
// `ckernel_sfpu_topk_xl.h`; the base TopK XL wrappers above and below are
// otherwise unchanged.
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init(std::uint32_t chunk_base) {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_, chunk_base);
}

template <std::uint32_t chunk_base_upper16>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_upper(std::uint32_t chunk_base_low16) {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_upper_<chunk_base_upper16>, chunk_base_low16);
}

template <std::uint32_t chunk_base_upper16, std::uint32_t chunk_base_lower16>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_static() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_static_<chunk_base_upper16, chunk_base_lower16>);
}

inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_reinit() {
    SAN_HOOK(unsupported());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_reinit_();
}
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split API.

template <std::uint32_t K, std::uint32_t group_id>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices(std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_<K, group_id>, dst_index, VectorMode::RC_custom);
}

// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split execution API.
template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major(std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_<K>, dst_index, VectorMode::RC_custom);
}

// Fused end-to-end: chunk-field-mask init + once-per-row global split.
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_global_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_);
}

template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_global(std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<K>, dst_index, VectorMode::RC_custom);
}

// Segmented fusion: per-segment split with a runtime segment base OR'd into
// the decoded index.
template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_global_base(
    std::uint32_t dst_index, std::uint32_t seg_base) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<K>, dst_index, VectorMode::RC_custom, seg_base);
}

template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_advance_chunk_base() {
    SAN_HOOK(unsupported());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_advance_chunk_base_<K>();
}
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split execution API.

// topk_large_indices: the op-specific survivor epilogue routines.

inline void llk_math_eltwise_unary_sfpu_topk_large_indices_reinit_fused_after_stamp() {
    SAN_HOOK(unsupported());
    ckernel::sfpu::_topk_large_indices_reinit_fused_after_stamp_();
}

inline void llk_math_eltwise_unary_sfpu_topk_large_indices_mark_neginf_indices_init() {
    SAN_HOOK(unsupported());
    ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_();
}

template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_large_indices_mark_neginf_indices(
    std::uint32_t dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>, dst_index, vector_mode);
}

// For a caller that has already opened the split and owns its own Dst addressing, so the dst-indexed
// params wrapper above would re-point it.
template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_large_indices_mark_neginf_indices_at_split() {
    SAN_HOOK(unsupported());
    // Row offset 0: the split survivor always sits at the base of Dst.
    ckernel::sfpu::_topk_xl_split_begin_(0);
    ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>();
    // Leave the Dst RWC where the next SFPU body expects it, as _topk_xl_separate_indices_* does.
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
}

template <std::uint32_t K>
inline void llk_math_eltwise_unary_sfpu_topk_large_indices_move_segment_result_at_split() {
    static_assert(K == 2048, "the segment-result move is hard-wired to K = 2048");
    SAN_HOOK(unsupported());
    // Row offset 0: the accumulator always sits at the base of Dst. The routine also needs ADDR_MOD_0 left
    // at +2 rows, which the global init in split_indices sets.
    ckernel::sfpu::_topk_xl_split_begin_(0);
    ckernel::sfpu::_topk_large_indices_move_segment_result_();
    // Leave the Dst RWC where the next SFPU body expects it, as _topk_xl_separate_indices_* does.
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
}

}  // namespace ckernel
