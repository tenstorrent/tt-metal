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

    // TODO: Expand programmability in order to support the dest dvalid scheme with different clients
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    const DataFormat srca_format = static_cast<DataFormat>(unpack_dst_format[srca_operand_id]);
    const DataFormat srcb_format = static_cast<DataFormat>(unpack_dst_format[srcb_operand_id]);

    // Determine the dest format based on the srcA/B formats and EN_32BIT_DEST_FORMAT
    if (EN_32BIT_DEST_FORMAT && is_src_fmt_fp32_dest_compatible(srca_format) &&
        is_src_fmt_fp32_dest_compatible(srcb_format)) {
        _llk_math_srcAB_hw_configure_<
            EN_IMPLIED_MATH_FORMAT,
            true /*EN_FP32_DEST_FORMAT*/,
            false /*EN_INT32_DEST_FORMAT*/>(srca_format, srcb_format);
    } else if (
        EN_32BIT_DEST_FORMAT && is_src_fmt_int32_dest_compatible(srca_format) &&
        is_src_fmt_int32_dest_compatible(srcb_format)) {
        _llk_math_srcAB_hw_configure_<
            EN_IMPLIED_MATH_FORMAT,
            false /*EN_FP32_DEST_FORMAT*/,
            true /*EN_INT32_DEST_FORMAT*/>(srca_format, srcb_format);
    } else {
        _llk_math_srcAB_hw_configure_<
            EN_IMPLIED_MATH_FORMAT,
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
