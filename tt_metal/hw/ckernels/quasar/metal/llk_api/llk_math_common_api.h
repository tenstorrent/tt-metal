// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
 * @brief Determines whether the source register format and Float32 destination format are a supported combination
 *
 * @param src_reg_fmt: The source register format
 */
inline bool is_src_fmt_fp32_dest_compatible(const DataFormat src_reg_fmt) {
    return src_reg_fmt == DataFormat::Float16_b || src_reg_fmt == DataFormat::Float16 ||
           src_reg_fmt == DataFormat::Tf32;
}

/**
 * @brief Determines whether the source register format and Int32 destination format are a supported combination
 *
 * @param src_reg_fmt: The source register format
 */
inline bool is_src_fmt_int32_dest_compatible(const DataFormat src_reg_fmt) {
    return src_reg_fmt == DataFormat::Int8 || src_reg_fmt == DataFormat::UInt8;
}

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
    if (EN_32BIT_DEST_FORMAT && is_src_fmt_fp32_dest_compatible(srca_format) &&
        is_src_fmt_fp32_dest_compatible(srcb_format)) {
        // TODO: AM; hardcoding false for EN_IMPLIED_MATH_FORMAT for now, will be fixed in issue #37720
        _llk_math_srcAB_hw_configure_<
            false /*EN_IMPLIED_MATH_FORMAT*/,
            true /*EN_FP32_DEST_FORMAT*/,
            false /*EN_INT32_DEST_FORMAT*/>(srca_format, srcb_format);
    } else if (
        EN_32BIT_DEST_FORMAT && is_src_fmt_int32_dest_compatible(srca_format) &&
        is_src_fmt_int32_dest_compatible(srcb_format)) {
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
 */
inline void llk_math_dest_section_done() {
    _llk_math_dest_section_done_<DST_SYNC_MODE>();
}

/**
 * @brief Initializes math–pack synchronization for the destination register.
 * Waits for any previous packs to finish, resets the dest bank id, initializes the MATH_PACK semaphore
 */
inline void llk_math_pack_sync_init() { _llk_math_pack_sync_init_<DST_SYNC_MODE>(); }
