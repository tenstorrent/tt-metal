// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_globals.h"
#include "internal/circular_buffer_interface.h"
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
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile,
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
            // FIXME: Need to support pack-untilize?
            static_assert(pack_addr_mode != PackMode::Untilize, "Use llk_pack_untilize APIs for pack-untilize.");
            // std::uint16_t out_tile_index =
            // (get_local_cb_interface(output_id).ublock_tile_cnt/get_local_cb_interface(output_id).ublock_ct)*get_local_cb_interface(output_id).row_tile_dim
            // +
            //                                 get_local_cb_interface(output_id).ublock_tile_cnt%get_local_cb_interface(output_id).ublock_ct;
            //                                 //FIXME: optimize perf
            // pack_tile_addr = get_local_cb_interface(output_id).fifo_wr_ptr +
            // get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1; pack_tile_addr +=
            // out_tile_index*(std::uint32_t)(get_local_cb_interface(output_id).fifo_page_size);

            // get_local_cb_interface(output_id).ublock_tile_cnt++;

            // if (get_local_cb_interface(output_id).ublock_tile_cnt ==
            // get_local_cb_interface(output_id).ublock_tile_dim) {
            //    get_local_cb_interface(output_id).ublock_tile_cnt=0;
            //    get_local_cb_interface(output_id).fifo_wr_tile_ptr +=
            //    (std::uint32_t)(get_local_cb_interface(output_id).fifo_page_size)*get_local_cb_interface(output_id).ublock_ct;
            // }
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
inline void llk_init_packer_dest_offset_registers(const std::uint32_t pack_output = 16) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_init_packer_dest_offset_registers_<DST_SYNC_MODE, pack_mode, diagonal>(face_r_dim, narrow_tile);
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_dest_init(const std::uint32_t pack_output = 16) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(face_r_dim, narrow_tile);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format_disaggregated(
    const std::uint32_t new_output, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4) {
    const std::uint32_t output_id = get_output_id(new_output);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile);
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

TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    _llk_pack_relu_config_(relu_config);
}

inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) { _llk_pack_reconfig_l1_acc_(enable); }
