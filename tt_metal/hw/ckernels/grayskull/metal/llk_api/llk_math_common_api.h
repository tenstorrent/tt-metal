// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_math_common.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "debug/waypoint.h"

/*************************************************************************
 * LLK MATH COMMON
 *************************************************************************/
template <bool untilize_en = false>
inline void llk_math_hw_configure_disaggregated() { /*Unused for GS*/ }

inline void llk_math_wait_for_dest_available() {
    WAYPOINT("MWDW");
    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
    WAYPOINT("MWDD");
}

template <bool is_fp32_dest_acc_en = false /*not used*/>
inline void llk_math_dest_section_done() {
    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <bool is_fp32_dest_acc_en = false /*not used*/>
inline void llk_math_pack_sync_init() {
    _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <bool mail2math = true, bool mail2pack = true>
inline void llk_math_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t *p_tile) {
    _llk_math_get_tile_<mail2math, mail2pack>(tile_index, p_tile);
}

template <bool mail2math = true, bool mail2pack = true>
inline void llk_math_release_tile(std::uint32_t operand) {
    _llk_math_release_tile_<mail2math, mail2pack>();
}

inline void llk_math_debug_dump(std::uint8_t *data, std::uint32_t byte_size) { _llk_math_debug_dump_(data, byte_size); }

inline void llk_math_debug_dump_seek(std::uint8_t offset) { _llk_math_debug_dump_seek_(offset); }

//The following functions are only needed for WHB0, they call empty functions for GS
inline void llk_math_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    _llk_math_reconfig_data_format_srca_();
}

inline void llk_math_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    _llk_math_reconfig_data_format_srcb_();
}

inline void llk_math_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    _llk_math_reconfig_data_format_();
}

inline void llk_math_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    _llk_math_reconfig_data_format_();
}

inline void llk_math_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    _llk_math_reconfig_data_format_srca_();
}

inline void llk_math_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    _llk_math_reconfig_data_format_srcb_();
}
