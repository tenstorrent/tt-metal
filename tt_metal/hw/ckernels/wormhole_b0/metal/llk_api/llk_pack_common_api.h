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

// Dest-acc mode a mid-kernel toggle left in effect, for the packer's data-format reconfig. Until the
// first llk_pack_wait_fp32_dest_acc the reconfig programs the flag it is instantiated with; from then on
// it reprograms the mode-dependent fields for the mode the toggle set. llk_pack_hw_configure clears the
// override: a compute_kernel_hw_startup returns the packer to the kernel-wide mode. (The mode is passed
// in, not sampled from ALU_ACC_CTRL_Fp32_enabled: a sample after the handshake could race MATH's next
// write of that bit, e.g. a fast-tilize init, and a sample at every reconfig could race MATH's startup
// write.)
inline bool pack_fp32_dest_acc_overridden = false;
inline bool pack_fp32_dest_acc_live = false;

/**
 * End a mid-kernel dest-acc override on the pack thread: later llk_pack_reconfig_data_format calls program
 * the kernel-wide mode again. On Wormhole the only path that reprograms the packer and ALU_ACC_CTRL for the
 * kernel-wide mode is llk_pack_hw_configure, which calls it.
 */
inline void llk_pack_clear_fp32_dest_acc_override() { pack_fp32_dest_acc_overridden = false; }

/**
 * Program the packer fields that the dest-acc mode decides for an output, the way configure_pack<mode>
 * programs them: PCK_DEST_RD_CTRL Read_32b_data / Read_int8 (the toggle's derivation), the exponent
 * threshold (fp32 Dest -> BFP-A underflow workaround) and Round_10b_mant (fp32 Dest -> Float16 source).
 *
 * @param enable    The dest-acc mode in effect: true for a 32-bit Dest, false for a 16-bit one.
 * @param output_id Output operand index the packer is configured for.
 */
// Cold path (toggles and reconfigs inside an override only): out of line, so the reconfig call sites share
// one copy, like reconfig_packer_data_format.
__attribute__((noinline)) inline void llk_pack_apply_fp32_dest_acc_override(
    const bool enable, const std::uint32_t output_id) {
    const std::uint32_t pack_src = pack_src_format[output_id];
    const std::uint32_t pack_dst = pack_dst_format[output_id];
    fp32_dest_acc::write_dest_rd_ctrl(enable, pack_src);
    if (enable) {
        ckernel::packer::reconfigure_exp_threshold<true>(pack_dst);
    } else {
        ckernel::packer::reconfigure_exp_threshold<false>(pack_dst);
    }
    const bool round_10b_mant = enable && pack_src == to_underlying(DataFormat::Float16);
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Round_10b_mant_RMW>(round_10b_mant);
}

/**
 * Pack-thread half of a mid-kernel FP32 dest-acc reconfiguration.
 *
 * Drains the packer FIFO, hands MATH the pack source format of `ocb` (MATH derives PCK_DEST_RD_CTRL
 * Read_32b_data / Read_int8 from it and the new mode), waits for MATH to program dest-acc CFG, then
 * records the mode so that later llk_pack_reconfig_data_format calls keep the packer in step with it,
 * and programs the packer fields that the mode decides for `ocb` (llk_pack_apply_fp32_dest_acc_override).
 *
 * @param ocb    Output circular buffer the packer is configured for (asserted against the packer config).
 * @param enable The mode MATH programs: true for a 32-bit Dest, false for a 16-bit one.
 * @note Must be called together with llk_unpack_wait_fp32_dest_acc and llk_math_set_fp32_dest_acc.
 */
inline void llk_pack_wait_fp32_dest_acc(const std::uint32_t ocb, const bool enable) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(ocb);
    const std::uint32_t pack_src = pack_src_format[output_id];
    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src, pack_dst_format[output_id]));
    _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>(static_cast<DataFormat>(pack_src));
    pack_fp32_dest_acc_live = enable;
    pack_fp32_dest_acc_overridden = true;
    // MATH has written Read_32b_data / Read_int8 to the same values; the override is applied whole so that
    // this is the one place the pack thread programs it.
    llk_pack_apply_fp32_dest_acc_override(enable, output_id);
}

/**
 * Configure the packer hardware for the given output operand.
 *
 * Face geometry (face_r_dim, num_faces), partial-face flag, narrow-tile flag and tile size are
 * derived from the output CB metadata associated with the operand id. Relu is left disabled here.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @param  pack_output         Output circular buffer / operand index.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure(std::uint32_t pack_output) {
    // A full configure programs the kernel-wide mode; a toggle's override ends here.
    llk_pack_clear_fp32_dest_acc_override();
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    SAN_HOOK(configure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
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
 * Face geometry (face_r_dim, num_faces), partial-face flag, narrow-tile flag and tile size are
 * derived from the new output's CB metadata. The mode-dependent fields follow the dest-acc mode in
 * effect: the kernel-wide flag, or the mode a mid-kernel enable_fp32_dest_acc / disable_fp32_dest_acc /
 * set_fp32_dest_acc left (pack_fp32_dest_acc_live), reprogrammed after the reconfig so that the
 * kernel-wide instantiation of the reconfig is the only one in the kernel.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register (kernel-wide mode).
 * @param  new_output          New output circular buffer / operand index to configure for.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);
    const bool fp32_dest_acc_en = pack_fp32_dest_acc_overridden ? pack_fp32_dest_acc_live : is_fp32_dest_acc_en;

    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
        StateVal<Operand<Exu::Pack>::NarrowTile>(narrow_tile),
        StateDiscard<std::uint32_t>(get_local_cb_interface(output_id).fifo_page_size)));
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile);
    if (fp32_dest_acc_en != is_fp32_dest_acc_en) {
        // A toggle moved the mode away from the kernel-wide flag: reprogram the fields the mode decides.
        llk_pack_apply_fp32_dest_acc_override(fp32_dest_acc_en, output_id);
    }
}

/**
 * Conditionally reconfigure the packer when switching output operands: only reprograms the data
 * format when the new output's destination format differs from the old one and neither is Invalid.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register (kernel-wide mode;
 *                             after a mid-kernel toggle the one-argument overload uses the toggled mode).
 * @param  old_output          Currently configured output operand index.
 * @param  new_output          New output operand index to switch to.
 */
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
