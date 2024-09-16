// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

/**
 * Helper function to reconfigure math srca and srcb data formats.
 */
ALWI void math_reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    MATH(( llk_math_reconfig_data_format(srca_new_operand, srcb_new_operand) ));
}

/**
 * Helper function to reconfigure srca/srcb data formats, only if they differ from existing formats.
*/
ALWI void math_reconfig_data_format(const uint32_t srca_old_operand, const uint32_t srca_new_operand, const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    MATH(( llk_math_reconfig_data_format(srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand) ));
}

/**
 * Helper function to reconfigure math srca data format.
 */
ALWI void math_reconfig_data_format_srca(const uint32_t srca_new_operand) {
    MATH(( llk_math_reconfig_data_format_srca(srca_new_operand) ));
}

/**
 * Helper function to reconfigure math srca input data format, only if it differs from existing format.
 */
ALWI void math_reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    MATH(( llk_math_reconfig_data_format_srca(srca_old_operand, srca_new_operand) ));
}

/**
 * Helper function to reconfigure math srcb input data format.
 */
ALWI void math_reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    MATH(( llk_math_reconfig_data_format_srcb(srcb_new_operand) ));
}

/**
 * Helper function to reconfigure math srcb input data format, only if it differs from existing format.
 */
ALWI void math_reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    MATH(( llk_math_reconfig_data_format_srcb(srcb_old_operand, srcb_new_operand) ));
}

}
