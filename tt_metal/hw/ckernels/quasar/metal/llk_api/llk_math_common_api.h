// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_math_common.h"
#include "llk_operands.h"

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
 **/
template <std::uint8_t SET_DEST_DVALID>
inline void llk_math_set_dvalid() {
    _llk_math_set_dvalid_<SET_DEST_DVALID>();
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on destination register using semaphores.
 *
 * The following functions should be phased out once the dest dvalid scheme is introduced
 */
// TODO: AM; move from semaphores to a per op programmable dest dvalid scheme, issue #37468

/**
 * @brief Waits until destination register space is available.
 * Blocks on the MATH_PACK semaphore until the packer gets the semaphore.
 */
inline void llk_math_wait_for_dest_available() {
    WAYPOINT("MWDW");
    _llk_math_wait_for_dest_available_();
    WAYPOINT("MWDD");
}

/**
 * @brief Signals that the current destination section is done.
 * After math is done, posts to the MATH_PACK semaphore so the packer can proceed;
 * @tparam EN_32BIT_DEST: Set to true to use 32bit math dest in Float32 or Int32 format
 */
template <bool EN_32BIT_DEST>
inline void llk_math_dest_section_done() {
    _llk_math_dest_section_done_<DST_SYNC_MODE, EN_32BIT_DEST>();
}

/**
 * @brief Initializes math–pack synchronization for the destination register.
 * Waits for any previous packs to finish, resets the dest bank id, initializes the MATH_PACK semaphore
 */
inline void llk_math_pack_sync_init() { _llk_math_pack_sync_init_<DST_SYNC_MODE>(); }

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
