// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "api/debug/waypoint.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_dest_clear_api.h"
#include "llk_fp32_dest_acc.h"
#include "llk_math_common.h"
#include "llk_operands.h"

// Need to revisit why we even need this
#define EPS 1.19209e-07  // std::numeric_limits::epsilon() for FP32

/*************************************************************************
 * LLK MATH COMMON
 *************************************************************************/
template <bool is_fp32_dest_acc_en>
inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand) {
    std::uint32_t srca_operand_id = get_operand_id(srca_operand);
    std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);
    SAN_HOOK(configure(
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[srca_operand_id]),
        StateVal<Operand<Exu::Sfpu>::Format>(unpack_dst_format[srcb_operand_id])));
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(
        unpack_dst_format[srca_operand_id], unpack_dst_format[srcb_operand_id]);
}

/**
 * Math-thread half of a mid-kernel FP32 dest-acc reconfiguration.
 *
 * Programs ALU_ACC_CTRL and PCK_DEST_RD_CTRL after Unpack and Pack have drained.
 *
 * @param enable True to enable FP32 dest accumulation, false to disable.
 * @note Must be called together with llk_unpack_wait_fp32_dest_acc and llk_pack_wait_fp32_dest_acc.
 */
inline void llk_math_set_fp32_dest_acc(bool enable) {
    SAN_HOOK(unsupported());
    _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(enable);
}

inline void llk_math_reconfig_remap(const bool remap_enable) {
    SAN_HOOK(unsupported());
    _llk_math_reconfig_remap_(remap_enable);
}

// math_clears_dest: the pack released this section without clearing it, so the math thread clears it here.
template <bool math_clears_dest = false>
inline void llk_math_wait_for_dest_available() {
    static_assert(
        !math_clears_dest || !llk_unpack_may_write_dest(),
        "math_clears_dest needs the math thread as DEST's only writer: no 32-bit unpack source in the kernel");
    WAYPOINT("MWDW");
    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
    if constexpr (math_clears_dest) {
        _llk_math_clear_dest_section_<DST_SYNC_MODE, DST_ACCUM_MODE>();
    }
    WAYPOINT("MWDD");
}

// Ends a run of math-cleared sections: once the pack has released the last one, all of DEST is clear again, as the
// default acquire expects.
inline void llk_math_dest_clear_region_end() {
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    _llk_math_clear_dest_section_<DstSync::SyncFull, DST_ACCUM_MODE>();
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_dest_section_done() {
    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_pack_sync_init() {
    _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    SAN_HOOK(reconfigure(StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[new_srca_operand_id])));
    LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, skip_int8>(unpack_dst_format[new_srca_operand_id]);
}

template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);
    SAN_HOOK(reconfigure(StateVal<Operand<Exu::Sfpu>::Format>(unpack_dst_format[new_srcb_operand_id])));
    LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, skip_int8>(unpack_dst_format[new_srcb_operand_id]);
}

template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[new_srca_operand_id]),
        StateVal<Operand<Exu::Sfpu>::Format>(unpack_dst_format[new_srcb_operand_id])));
    LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
    _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, skip_int8>(
        unpack_dst_format[new_srca_operand_id], unpack_dst_format[new_srcb_operand_id]);
}

template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
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
        llk_math_reconfig_data_format<is_fp32_dest_acc_en, skip_int8>(srca_new_operand, srcb_new_operand);
    } else if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id])) {
        llk_math_reconfig_data_format_srca<is_fp32_dest_acc_en, skip_int8>(srca_new_operand);
    } else if ((unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format_srcb<is_fp32_dest_acc_en, skip_int8>(srcb_new_operand);
    }
}

template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if ((unpack_dst_format[old_srca_operand_id] != unpack_dst_format[new_srca_operand_id])) {
        llk_math_reconfig_data_format_srca<is_fp32_dest_acc_en, skip_int8>(srca_new_operand);
    }
}

template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void llk_math_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if ((unpack_dst_format[old_srcb_operand_id] != unpack_dst_format[new_srcb_operand_id])) {
        llk_math_reconfig_data_format_srcb<is_fp32_dest_acc_en, skip_int8>(srcb_new_operand);
    }
}

/**
 * @brief Returns the effective math fidelity for an eltwise binary operation.
 * Math fidelity only applies to ELWMUL; for all other binary ops (ELWADD/ELWSUB), LoFi is used.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam math_fidelity: The requested math fidelity
 * @return The requested math_fidelity for ELWMUL, MathFidelity::LoFi otherwise.
 */
template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity>
inline constexpr MathFidelity get_effective_math_fidelity() {
    return (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? math_fidelity : MathFidelity::LoFi;
}
