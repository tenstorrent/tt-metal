// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/api.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "internal/circular_buffer_interface.h"
#include "llk_assert.h"
#include "llk_fp32_dest_acc.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

/*************************************************************************
 * LLK PACK COMMON
 *************************************************************************/

/**
 * Pack-thread half of a mid-kernel FP32 dest-acc reconfiguration.
 *
 * Drains the packer FIFO, waits for MATH to program dest-acc CFG (including PCK_DEST_RD_CTRL),
 * then STALLWAITs.
 *
 * @note Must be called together with llk_unpack_wait_fp32_dest_acc and llk_math_set_fp32_dest_acc.
 */
inline void llk_pack_wait_fp32_dest_acc(bool enable) {
    SAN_HOOK(unsupported());
    _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>(enable);
}

/**
 * Configure the packer hardware for the given output operand.
 *
 * Face geometry (face_r_dim, num_faces), narrow-tile flag and tile size are derived from the output
 * CB metadata associated with the operand id. The packer's partial-face flag follows the face height
 * (faces shorter than FACE_R_DIM rows share one block-float exponent chunk), not the CB's tile-height
 * partial_face flag, as in llk_pack_reconfig_data_format; a block-float output with short faces must
 * fit face_r_dim * num_faces <= FACE_R_DIM (asserted). Relu is left disabled here.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @param  pack_output         Output circular buffer / operand index.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    // The packer's partial-face flag selects one exponent chunk per block-float tile, which is right only for
    // faces shorter than FACE_R_DIM rows; the CB's partial_face flag (tile height below the full tile) is also
    // set for a 16x32 tile whose two full-height faces need a chunk each. Same rule as the data-format reconfig.
    const bool partial_face = face_r_dim < FACE_R_DIM;
    LLK_ASSERT(
        !IS_BFP_FORMAT(pack_dst_format[output_id]) || face_r_dim == FACE_R_DIM || face_r_dim * num_faces <= FACE_R_DIM,
        "short-face block-float tile needs more than one exponent chunk");
    const bool narrow_tile = get_output_narrow_tile(output_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    SAN_HOOK(configure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        // The sanitizer tracks the CB's partial_face flag, as the pack init / execute hooks do.
        StateVal<Operand<Exu::Pack>::PartialFace>(get_output_partial_face(output_id)),
        StateVal<Operand<Exu::Pack>::NarrowTile>(narrow_tile),
        StateDiscard<std::uint32_t>(tile_size)));
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

/**
 * Compute the L1 write address for a packer output tile.
 *
 * For out-of-order output, the address is derived from the explicit tile index. For in-order
 * output, the running fifo write-tile pointer is used and then advanced by one page. Only
 * PackMode::Default and PackMode::Untilize are accepted; pack-untilize must use the dedicated
 * llk_pack_untilize APIs.
 *
 * @tparam out_of_order_output When true, address by output_tile_index; otherwise use the running fifo pointer.
 * @tparam pack_addr_mode      Packer addressing mode (Default or Untilize).
 * @param  output_id           Resolved output operand id.
 * @param  output_tile_index   Tile index within the output (used only for out-of-order output).
 * @return L1 write address for the requested output tile.
 */
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

/**
 * Block the packer until the math thread signals that the destination register is ready to pack.
 */
inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

/**
 * Signal that the packer has finished its current destination-register section, releasing it
 * back to the math thread.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_dest_section_done() {
    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

/**
 * Initialize the packer destination-offset registers for the given pack mode.
 *
 * Face row dimension and narrow-tile flag are derived from the output CB metadata.
 *
 * @tparam pack_mode   Packer program mode.
 * @tparam diagonal    Whether to use diagonal packing.
 * @param  pack_output Output circular buffer / operand index.
 */
template <PackMode pack_mode = PackMode::Default, bool diagonal = false>
inline void llk_init_packer_dest_offset_registers(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_init_packer_dest_offset_registers_<DST_SYNC_MODE, pack_mode, diagonal>(face_r_dim, narrow_tile);
}

/**
 * Initialize the packer destination register for the given pack mode.
 *
 * Face row dimension and narrow-tile flag are derived from the output CB metadata.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam pack_mode           Packer program mode.
 * @param  pack_output         Output circular buffer / operand index.
 */
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_dest_init(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(face_r_dim, narrow_tile);
}

/**
 * Reconfigure the packer for a new output operand's data format.
 *
 * Face geometry (face_r_dim, num_faces), narrow-tile flag and tile size are derived from the new
 * output's CB metadata. The packer's partial-face flag follows the face height: faces shorter than
 * FACE_R_DIM rows make a block-float tile carry one exponent chunk in total, full-height faces one
 * chunk per face. That single chunk holds at most FACE_R_DIM exponent rows, so a block-float output
 * with short faces is supported only while face_r_dim * num_faces <= FACE_R_DIM (the host's default
 * face shapes satisfy this; a block-float 16x32 tile of four 8-row faces would not; other formats
 * have no exponent section and are unrestricted). The CB's own partial_face flag (tile height
 * below the full tile) is not used here, since it is also set for a 16x32 tile whose two full-height
 * faces need a chunk each.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @param  new_output          New output circular buffer / operand index to configure for.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = face_r_dim < FACE_R_DIM;
    const bool narrow_tile = get_output_narrow_tile(output_id);
    LLK_ASSERT(
        !IS_BFP_FORMAT(pack_dst_format[output_id]) || face_r_dim == FACE_R_DIM || face_r_dim * num_faces <= FACE_R_DIM,
        "short-face block-float tile needs more than one exponent chunk");

    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        // The sanitizer tracks the CB's partial_face flag, as the other pack hooks do; partial_face above is
        // the packer's own exponent-chunk flag, which this hook does not model.
        StateVal<Operand<Exu::Pack>::PartialFace>(get_output_partial_face(output_id)),
        StateVal<Operand<Exu::Pack>::NarrowTile>(narrow_tile),
        StateDiscard<std::uint32_t>(get_local_cb_interface(output_id).fifo_page_size)));
    LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile);
}

/**
 * Conditionally reconfigure the packer when switching output operands: reprograms the packer when the
 * new output's data formats or tile geometry (face height, face count, narrow-tile flag) differ from
 * the old one's and neither destination format is Invalid. Face height and face count are compared
 * because the reconfig programs the block-float exponent section sizes from them. The narrow-tile
 * flag is the only term that changes on a same-format 16x32 -> 32x16 switch (same face height and
 * count). Both shapes use two exponent chunks; the comparison keeps the sanitizer's NarrowTile and
 * PartialFace operand state current for the next pack init.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @param  old_output          Currently configured output operand index.
 * @param  new_output          New output operand index to switch to.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    std::uint32_t old_output_id = get_output_id(old_output);
    std::uint32_t new_output_id = get_output_id(new_output);

    const bool format_changed = (pack_dst_format[old_output_id] != pack_dst_format[new_output_id]) ||
                                (pack_src_format[old_output_id] != pack_src_format[new_output_id]);
    const bool geometry_changed = (get_output_face_r_dim(old_output_id) != get_output_face_r_dim(new_output_id)) ||
                                  (get_output_num_faces(old_output_id) != get_output_num_faces(new_output_id)) ||
                                  (get_output_narrow_tile(old_output_id) != get_output_narrow_tile(new_output_id));

    if ((format_changed || geometry_changed) && (pack_dst_format[old_output_id] != (uint)DataFormat::Invalid) &&
        (pack_dst_format[new_output_id] != (uint)DataFormat::Invalid)) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(new_output);
    }
}

/**
 * Program the packer relu configuration register.
 *
 * @param relu_config Relu mode/threshold configuration.
 */
TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    SAN_HOOK(unsupported());
    _llk_pack_relu_config_(relu_config);
}

/**
 * Enable or disable packer L1 accumulation.
 *
 * @param enable Non-zero to enable L1 accumulation, zero to disable.
 */
inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) { _llk_pack_reconfig_l1_acc_(enable); }
