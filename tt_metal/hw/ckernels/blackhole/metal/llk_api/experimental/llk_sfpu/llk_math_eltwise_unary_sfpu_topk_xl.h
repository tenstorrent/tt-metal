// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#include "sanitizer/api.h"

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

// Reprogram only the MOP Expander after a topk_xl copy, instead of a full
// topk_xl_init. See ckernel_sfpu_topk_xl.h for what copy init clobbers.
template <bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_reinit_mop_after_copy() {
    SAN_HOOK(unsupported());
    ckernel::sfpu::topk_mop_config<fused>();
}

// Restore the ADDR_MODs and MOP state the unfused rebuild needs after a copy.
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

}  // namespace ckernel
