// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "llk_param_structs.h"

// Need to revisit why we even need this
#define EPS 1.19209e-07  // std::numeric_limits::epsilon() for FP32

/*************************************************************************
 * LLK MATH COMMON
 *************************************************************************/
template <bool is_fp32_dest_acc_en>
inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand) {
    std::uint32_t srca_operand_id = get_operand_id(srca_operand);
    std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(
        unpack_dst_format[srca_operand_id], unpack_dst_format[srcb_operand_id]);
}

inline void llk_math_reconfig_remap(const bool remap_enable) { _llk_math_reconfig_remap_(remap_enable); }

inline void llk_math_wait_for_dest_available() {
    WAYPOINT("MWDW");
    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
    WAYPOINT("MWDD");
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_dest_section_done() {
    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_pack_sync_init() {
    _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

inline void llk_math_debug_dump(std::uint8_t* data, std::uint32_t byte_size) { _llk_math_debug_dump_(data, byte_size); }

inline void llk_math_debug_dump_seek(std::uint8_t offset) { _llk_math_debug_dump_seek_(offset); }

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, to_from_int8>(unpack_dst_format[new_srca_operand_id]);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, to_from_int8>(unpack_dst_format[new_srcb_operand_id]);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, to_from_int8>(
        unpack_dst_format[new_srca_operand_id], unpack_dst_format[new_srcb_operand_id]);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id]) &&
        (unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format<is_fp32_dest_acc_en, to_from_int8>(srca_new_operand, srcb_new_operand);
    } else if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id])) {
        llk_math_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8>(srca_new_operand);
    } else if ((unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8>(srcb_new_operand);
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id])) {
        llk_math_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8>(srca_new_operand);
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if ((unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8>(srcb_new_operand);
    }
}

inline std::uint32_t llk_math_get_compute_special_value_flags() { return _llk_math_get_compute_special_value_flags_(); }

inline std::uint32_t llk_math_get_compute_special_value_flags_fpu(std::uint32_t special_value_flags_reg) {
    constexpr std::uint32_t special_value_flags_fpu_mask = 0xf;
    constexpr std::uint32_t special_value_flags_fpu_shift = 4;
    return (special_value_flags_reg & special_value_flags_fpu_mask) >> special_value_flags_fpu_shift;
}

inline std::uint32_t llk_math_get_compute_special_value_flags_sfpu(std::uint32_t special_value_flags_reg) {
    constexpr std::uint32_t special_value_flags_sfpu_mask = 0xf;
    constexpr std::uint32_t special_value_flags_sfpu_shift = 0;
    return (special_value_flags_reg & special_value_flags_sfpu_mask) >> special_value_flags_sfpu_shift;
}

inline void llk_math_clear_compute_special_value_flags() { _llk_math_clear_compute_special_value_flags_(); }

inline void llk_math_store_compute_special_value_flags_to_l1(std::uint32_t l1_addr) {
    volatile tt_l1_ptr std::uint32_t* l1_addr_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(l1_addr);
    l1_addr_ptr[0] = _llk_math_get_compute_special_value_flags_();
}
