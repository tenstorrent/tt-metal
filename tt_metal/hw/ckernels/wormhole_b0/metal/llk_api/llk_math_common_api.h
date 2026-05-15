// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "api/debug/waypoint.h"

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

inline void llk_math_set_fp32_dest_acc(bool enable) { _llk_math_set_fp32_dest_acc_(enable); }

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
