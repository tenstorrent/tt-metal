// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

/*************************************************************************
 * LLK PACK COMMON
 *************************************************************************/

inline void llk_pack_set_fp32_dest_acc(bool enable) { _llk_pack_set_fp32_dest_acc_(enable); }

template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        0 /*relu_config*/);
}

template <bool out_of_order_output, PackMode pack_addr_mode = PackMode::Default>
inline std::uint32_t get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index) {
    static_assert(
        pack_addr_mode == PackMode::Default || pack_addr_mode == PackMode::Untilize,
        "Pack tile address helper supports PackMode::Default and PackMode::Untilize only");
    std::uint32_t pack_tile_addr;
    if constexpr (out_of_order_output) {
        pack_tile_addr = get_local_cb_interface(output_id).fifo_wr_ptr +
                         (std::uint32_t)(get_local_cb_interface(output_id).fifo_page_size) * output_tile_index - 1;
    } else {
        if constexpr (pack_addr_mode == PackMode::Untilize) {
            static_assert(pack_addr_mode != PackMode::Untilize, "Use llk_pack_untilize APIs for pack-untilize.");
            // TODO: uplift this option from BBE
        } else {
            pack_tile_addr =
                get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
            get_local_cb_interface(output_id).fifo_wr_tile_ptr += get_local_cb_interface(output_id).fifo_page_size;
        }
    }
    return pack_tile_addr;
}

inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

template <bool is_fp32_dest_acc_en>
inline void llk_pack_dest_section_done() {
    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <PackMode pack_mode = PackMode::Default, bool diagonal = false>
inline void llk_init_packer_dest_offset_registers([[maybe_unused]] const std::uint32_t pack_output = 16) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Blackhole llk_init_packer_dest_offset_registers: PackMode::Tilize is not used on this path");
    (void)diagonal;
    _llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>();
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_dest_init([[maybe_unused]] const std::uint32_t pack_output = 16) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Blackhole llk_pack_dest_init: PackMode::Tilize is not used on this path");
    _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        false /* partial_face */);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format_disaggregated(
    const std::uint32_t new_output, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4) {
    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);

    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        false);  // partial_face
}

// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    std::uint32_t old_output_id = get_output_id(old_output);
    std::uint32_t new_output_id = get_output_id(new_output);

    if ((pack_dst_format[old_output_id] != pack_dst_format[new_output_id]) &&
        (pack_dst_format[old_output_id] != (uint)DataFormat::Invalid) &&
        (pack_dst_format[new_output_id] != (uint)DataFormat::Invalid)) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(new_output);
    }
}

TT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) { _llk_pack_relu_config_(config); }

inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) { _llk_pack_reconfig_l1_acc_(enable); }
