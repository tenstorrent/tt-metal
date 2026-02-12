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
// template <bool EN_IMPLIED_MATH_FORMAT, bool EN_FP32_MATH_FORMAT, bool EN_INT32_MATH_FORMAT>
// inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand) {
//     std::uint32_t srca_operand_id = get_operand_id(srca_operand);
//     std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);

//     set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>(
//         {dest_dvalid_client::FPU, dest_dvalid_client::PACK});  // incomplete
//     _llk_math_srcAB_hw_configure_<EN_IMPLIED_MATH_FORMAT, EN_FP32_MATH_FORMAT, EN_INT32_MATH_FORMAT>(
//         static_cast<DataFormat>(unpack_dst_format[srca_operand_id]),
//         static_cast<DataFormat>(unpack_dst_format[srcb_operand_id]));
// }

// inline void llk_math_pack_sync_init() { _llk_math_pack_sync_init_<DST_SYNC_MODE>(); }

// template <uint8_t SET_DEST_DVALID>
// inline void llk_math_set_dvalid() {
//     _llk_math_set_dvalid_<SET_DEST_DVALID>();
// }
