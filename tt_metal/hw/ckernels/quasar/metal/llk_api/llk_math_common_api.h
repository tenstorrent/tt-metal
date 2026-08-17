// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "llk_dest_dvalid.h"

/*************************************************************************
 * LLK MATH COMMON
 *************************************************************************/

/**
 *
 * @brief Configures math hardware.
 * Sets up ALU formats for math destination register and source registers.
 *
 * @tparam EN_IMPLIED_MATH_FORMAT: If set to true, will imply math dest format from SrcA reg format
 * @tparam EN_32BIT_DEST_FORMAT: Set to true to use 32bit math dest in Float32 or Int32 format
 * @param srca_operand: The srcA input operand circular buffer, used to infer srcA data_format if not implied math
 * format
 * @param srcb_operand: The srcB input operand circular buffer, used to infer srcB data_format if not implied math
 * format
 */
template <bool EN_32BIT_DEST_FORMAT>
inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_operand);
    const std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);

    const DataFormat srca_format = static_cast<DataFormat>(unpack_dst_format[srca_operand_id]);
    const DataFormat srcb_format = static_cast<DataFormat>(unpack_dst_format[srcb_operand_id]);

    // TODO: AM; introduce dest mode enum, issue #37483
    // Determine the dest format based on the srcA/B formats and EN_32BIT_DEST_FORMAT
    if (EN_32BIT_DEST_FORMAT && _is_src_fmt_fp32_dest_compatible_(srca_format) &&
        _is_src_fmt_fp32_dest_compatible_(srcb_format)) {
        // TODO: AM; hardcoding false for EN_IMPLIED_MATH_FORMAT for now, will be fixed in issue #37720
        _llk_math_srcAB_hw_configure_<
            false /*EN_IMPLIED_MATH_FORMAT*/,
            true /*EN_FP32_DEST_FORMAT*/,
            false /*EN_INT32_DEST_FORMAT*/>(srca_format, srcb_format);
    } else if (
        EN_32BIT_DEST_FORMAT && _is_src_fmt_int32_dest_compatible_(srca_format) &&
        _is_src_fmt_int32_dest_compatible_(srcb_format)) {
        // TODO: AM; hardcoding false for EN_IMPLIED_MATH_FORMAT for now, will be fixed in issue #37720
        _llk_math_srcAB_hw_configure_<
            false /*EN_IMPLIED_MATH_FORMAT*/,
            false /*EN_FP32_DEST_FORMAT*/,
            true /*EN_INT32_DEST_FORMAT*/>(srca_format, srcb_format);
    } else {
        // TODO: AM; hardcoding false for EN_IMPLIED_MATH_FORMAT for now, will be fixed in issue #37720
        _llk_math_srcAB_hw_configure_<
            false /*EN_IMPLIED_MATH_FORMAT*/,
            false /*EN_FP32_DEST_FORMAT*/,
            false /*EN_INT32_DEST_FORMAT*/>(srca_format, srcb_format);
    }
}

inline void llk_math_reconfig_remap(const bool /*remap_enable*/) {}

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

/**
 * @brief Sets the dest dvalid for FPU/SFPU
 *
 * @tparam SET_DEST_DVALID: which client to set data valid for, values = p_cleardvalid::FPU/SFPU
 * @tparam DST: Destination register banking mode: SyncHalf = double banked (math/pack overlap), SyncFull = one bank
 *(serialized)
 *
 * @warning SYNC SCHEME: dest-dvalid. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 **/
template <std::uint8_t SET_DEST_DVALID, DstSync DST>
inline void llk_math_set_dvalid() {
    _llk_math_set_dvalid_<SET_DEST_DVALID, DST>();
}

/**
 * @brief Waits until destination register space is available.
 * The FPU→PACK dvalid ring gates the math coprocessor thread via CLEARDVALID;
 * no explicit software wait is needed here.
 */
inline void llk_math_wait_for_dest_available() {}

/**
 * @brief Signals that the current destination section is done.
 * Pulses the FPU dvalid client (waits for the ring condition, then toggles)
 * so the pack client can proceed.  Uses a single pulse — the double-pulse
 * bank-reset form deadlocks inside a configured ring.
 * @tparam EN_32BIT_DEST: Set to true to use 32bit math dest in Float32 or Int32 format
 */
template <bool EN_32BIT_DEST>
inline void llk_math_dest_section_done() {
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::FPU, 0);
    if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
        _update_dest_register_offset_<EN_32BIT_DEST>();
        std::uint32_t base_addr = _get_dest_buffer_base_();
        TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::MATH, p_stall::WAIT_SFPU);
        _set_dest_section_base_<ckernel::TRISC_ID>(base_addr);
    }
}

/**
 * @brief Initializes the FPU side of the dest-dvalid ring and resets the dest section base.
 * Configures FPU as the ring-start client of an {FPU, PACK} dvalid ring
 * with auto bank-ID toggle disabled (software bank management).
 */
inline void llk_math_pack_sync_init() {
    _reset_dest_register_offset_();
    _set_dest_section_base_<ckernel::TRISC_ID>(_get_dest_buffer_base_());
    _llk_dest_dvalid_exclude_<dest_dvalid_client::UNPACK>();
    _llk_dest_dvalid_exclude_<dest_dvalid_client::SFPU>();
    _llk_dest_dvalid_enable_<dest_dvalid_client::FPU>();
    cfg_rmw(MATH_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_RMW, 1);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

// Math has no per-tile data-format state on Quasar; format reconfig is unpack-only.
// The wrappers below are intentionally empty no-ops, kept so reconfig_data_format.h
// can issue MATH((...)) uniformly across arches.
template <[[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srca(const std::uint32_t /*srca_new_operand*/) {}

template <[[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srcb(const std::uint32_t /*srcb_new_operand*/) {}

template <[[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_math_reconfig_data_format(
    const std::uint32_t /*srca_new_operand*/, const std::uint32_t /*srcb_new_operand*/) {}

template <[[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_math_reconfig_data_format(
    const std::uint32_t /*srca_old_operand*/,
    const std::uint32_t /*srca_new_operand*/,
    const std::uint32_t /*srcb_old_operand*/,
    const std::uint32_t /*srcb_new_operand*/) {}

template <[[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srca(
    const std::uint32_t /*srca_old_operand*/, const std::uint32_t /*srca_new_operand*/) {}

template <[[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_math_reconfig_data_format_srcb(
    const std::uint32_t /*srcb_old_operand*/, const std::uint32_t /*srcb_new_operand*/) {}
