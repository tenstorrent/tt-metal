// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/unpack.h"
#include "compute_kernel_api/math.h"

namespace ckernel {

/**
 * Helper function to reconfigure srca and srcb data formats.
 */
template <bool float_only = true, bool reconfig_math = true>
ALWI void reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    unpack_reconfig_data_format<float_only>(srca_new_operand, srcb_new_operand);
    if constexpr (reconfig_math) {
        math_reconfig_data_format<float_only>(srca_new_operand, srcb_new_operand);
    }
}

/**
 * Helper function to reconfigure srca/srcb data formats, only if they differ from existing formats.
*/
template <bool float_only = true, bool reconfig_math = true>
ALWI void reconfig_data_format(const uint32_t srca_old_operand, const uint32_t srca_new_operand, const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    unpack_reconfig_data_format<float_only>(srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand);
    if constexpr (reconfig_math) {
        math_reconfig_data_format<float_only>(srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand);
    }
}

/**
 * Helper function to reconfigure srca data format.
 */
template <bool float_only = true, bool reconfig_math = true>
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    unpack_reconfig_data_format_srca<float_only>(srca_new_operand);
    if constexpr (reconfig_math) {
        math_reconfig_data_format_srca<float_only>(srca_new_operand);
    }
}

/**
 * Helper function to reconfigure srca input data format, only if it differs from existing format.
 */
template <bool float_only = true, bool reconfig_math = true>
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    unpack_reconfig_data_format_srca<float_only>(srca_old_operand, srca_new_operand);
    if constexpr (reconfig_math) {
        math_reconfig_data_format_srca<float_only>(srca_old_operand, srca_new_operand);
    }
}

/**
 * Helper function to reconfigure srcb input data format.
 */
template <bool float_only = true, bool reconfig_math = true>
ALWI void econfig_data_format_srcb(const uint32_t srcb_new_operand) {
    unpack_reconfig_data_format_srcb<float_only>(srcb_new_operand);
    if constexpr (reconfig_math) {
        math_reconfig_data_format_srcb<float_only>(srcb_new_operand);
    }
}

/**
 * Helper function to reconfigure srcb input data format, only if it differs from existing format.
 */
template <bool float_only = true, bool reconfig_math = true>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    unpack_reconfig_data_format_srcb<float_only>(srcb_old_operand, srcb_new_operand);
    if constexpr (reconfig_math) {
        math_reconfig_data_format_srcb<float_only>(srcb_old_operand, srcb_new_operand);
    }
}

}
