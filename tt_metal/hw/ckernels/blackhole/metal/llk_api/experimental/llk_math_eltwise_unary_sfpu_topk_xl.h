// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_topk_xl.h"

namespace ckernel {

template <uint32_t K, bool APPROXIMATE, bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_topk_xl_init_<K, fused>);
}

template <uint32_t K, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_local_sort(
    uint dst_index, bool ascending, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_local_sort_<K, APPROXIMATE>, dst_index, vector_mode, dst_index, ascending);
}

template <uint32_t K, bool APPROXIMATE, bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_merge(uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_merge_<K, APPROXIMATE, fused>, dst_index, vector_mode, dst_index);
}

template <uint32_t K, bool APPROXIMATE, bool fused>
inline void llk_math_eltwise_unary_sfpu_topk_xl_rebuild(
    uint dst_index, bool ascending, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_rebuild_<K, APPROXIMATE, fused>, dst_index, vector_mode, dst_index, ascending);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::_topk_xl_add_lsb_indices_init_);
}

template <uint32_t K, bool APPROXIMATE, uint32_t core_id>
inline void llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_add_lsb_indices_<K, APPROXIMATE, core_id>, dst_index, VectorMode::RC_custom);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values_init() {
    ckernel::sfpu::_topk_xl_remove_msb_values_init_();
}

template <uint32_t K, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values(uint dst_index) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH | p_stall::PACK);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::_topk_xl_remove_msb_values_<K>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_init(uint32_t group_id_bit_shift) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_init_, group_id_bit_shift);
}

// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split API.
// These wrappers expose the row-major SFPU helpers added to
// `ckernel_sfpu_topk_xl.h`; the base TopK XL wrappers above and below are
// otherwise unchanged.
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init(uint32_t chunk_base) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_, chunk_base);
}

template <bool APPROXIMATE, uint32_t chunk_base_upper16>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_upper(uint32_t chunk_base_low16) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_upper_<chunk_base_upper16>, chunk_base_low16);
}

template <bool APPROXIMATE, uint32_t chunk_base_upper16, uint32_t chunk_base_lower16>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_static() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_init_static_<chunk_base_upper16, chunk_base_lower16>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_reinit() {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_reinit_();
}
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split API.

template <uint32_t K, bool APPROXIMATE, uint32_t group_id>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_<K, APPROXIMATE, group_id>, dst_index, VectorMode::RC_custom);
}

// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split execution API.
template <uint32_t K, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_<K, APPROXIMATE>, dst_index, VectorMode::RC_custom);
}

template <uint32_t K, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_advance_chunk_base() {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_advance_chunk_base_<K>();
}
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split execution API.

}  // namespace ckernel
