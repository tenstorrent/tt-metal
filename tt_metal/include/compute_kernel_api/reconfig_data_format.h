// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

/**
 * Helper function to reconfigure srca and srcb data formats.
 */
template <bool to_from_int8 = false>
ALWI void reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(srca_new_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(srca_new_operand, srcb_new_operand)));
}

/**
 * Helper function to reconfigure srca/srcb data formats, only if they differ from existing formats.
 */
template <bool to_from_int8 = false>
ALWI void reconfig_data_format(
    const uint32_t srca_old_operand,
    const uint32_t srca_new_operand,
    const uint32_t srcb_old_operand,
    const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(
        srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(
        srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand)));
}

/**
 * Helper function to reconfigure srca data format.
 */
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_new_operand)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_new_operand)));
}

/**
 * Helper function to reconfigure srca input data format, only if it differs from existing format.
 */
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_old_operand, srca_new_operand)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_old_operand, srca_new_operand)));
}

/**
 * Helper function to reconfigure srcb input data format.
 */
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_new_operand)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_new_operand)));
}

/**
 * Helper function to reconfigure srcb input data format, only if it differs from existing format.
 */
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_old_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_old_operand, srcb_new_operand)));
}

}  // namespace ckernel
