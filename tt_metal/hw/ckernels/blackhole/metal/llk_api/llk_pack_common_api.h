// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/api.h"
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "llk_assert.h"
#include "llk_dest_clear_api.h"
#include "llk_fp32_dest_acc.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_size_mode.h"
#include "llk_pack_block_tile.h"
#include "llk_copy_kernel.h"

namespace ckernel::packer {
template <>
struct llk_pack_copy_api<1> {
    static constexpr bool value = LLK_COPY_KERNEL;
};
}  // namespace ckernel::packer

namespace ckernel::packer {
template <>
inline constexpr bool pack_size_mode<DST_SYNC_MODE> = LLK_SIZE_MODE;
template <>
struct llk_pack_in0_only32<1> {
    static constexpr bool value = DST_ACCUM_MODE && llk_pack_out32(unpack_src_format[0]) &&
                                  unpack_src_format[1] != static_cast<std::uint8_t>(DataFormat::Invalid) && !llk_pack_out32(unpack_src_format[1]);
};
}  // namespace ckernel::packer

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
    ckernel::packer::llk_pack_live_forget();
}

/**
 * Configure the packer hardware for the given output operand.
 *
 * Face geometry (face_r_dim, num_faces), tile column dimension, partial-face flag and tile
 * size are derived from the output CB metadata associated with the operand id. Relu is left
 * disabled here.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @param  pack_output         Output circular buffer / operand index.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);

    const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;

    SAN_HOOK(configure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::TileWidth>(tile_c_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
        StateDiscard<std::uint32_t>(tile_size)));
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default, LLK_SIZE_MODE>(
        pack_src_format[output_id],
        pack_dst_format[output_id],
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        0 /*relu_config*/);
    llk_pack_hw_configure_live<is_fp32_dest_acc_en>(face_r_dim, tile_c_dim, num_faces);
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

/**
 * Block the packer until the math thread signals that the destination register is ready to pack.
 */
inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

/**
 * Signal that the packer has finished its current destination-register section, releasing it
 * back to the math thread.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam math_clears_dest    Release without clearing; the math thread clears the section when it acquires it.
 * @tparam words               Release word form of _llk_pack_dest_section_done_; 0 is the full sequence.
 */
template <bool is_fp32_dest_acc_en, bool math_clears_dest = false, std::uint32_t words = 0>
inline void llk_pack_dest_section_done() {
    static_assert(
        !math_clears_dest || !llk_unpack_may_write_dest(),
        "math_clears_dest needs the math thread as DEST's only writer: no 32-bit unpack source in the kernel");
    if constexpr (!math_clears_dest && words == 0) {
        // main's two-argument form, which a kernel may wrap.
        _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
    } else {
        _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en, !math_clears_dest, words>();
    }
}

/**
 * Initialize the packer destination-offset registers for the given pack mode.
 *
 * @tparam pack_mode   Packer program mode (Default or Untilize; Tilize is not used on this path).
 * @tparam diagonal    Diagonal packing flag (unused on Blackhole).
 * @param  pack_output Output circular buffer / operand index.
 */
template <PackMode pack_mode = PackMode::Default, bool diagonal = false>
inline void llk_init_packer_dest_offset_registers([[maybe_unused]] const std::uint32_t pack_output) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Blackhole llk_init_packer_dest_offset_registers: PackMode::Tilize is not used on this path");
    (void)diagonal;
    _llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>();
}

/**
 * Initialize the packer destination register for the given pack mode.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam pack_mode           Packer program mode (Default or Untilize; Tilize is not used on this path).
 * @param  pack_output         Output circular buffer / operand index.
 */
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_dest_init([[maybe_unused]] const std::uint32_t pack_output) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Blackhole llk_pack_dest_init: PackMode::Tilize is not used on this path");
    _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

// A run-time output's reconfigure through the kernel's constant profiles: where the immediate reconfigure is allowed and no
// SKIP_ALL applies, the output's (input, output, face count) selects that profile's immediate sequence.
template <bool is_fp32_dest_acc_en, bool gate = true>
constexpr bool llk_pack_reconfig_table_ok() {
    constexpr ckernel::packer::llk_pack_reconfig_imm_t imm =
        ckernel::packer::llk_pack_reconfig_imm<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
            pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
    constexpr bool skip_all =
        (ckernel::packer::llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
             pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face) &
         ckernel::packer::LLK_PACK_RECONFIG_SKIP_ALL) != 0;
    constexpr ckernel::packer::llk_pack_reconfig_profiles_t prof =
        ckernel::packer::llk_pack_reconfig_profiles(pack_src_format, pack_dst_format, pack_tile_num_faces);
    return !LLK_SIZE_MODE && imm.ok && !skip_all && prof.n >= 1 && prof.n <= 8;
}

template <bool is_fp32_dest_acc_en, std::uint32_t k, bool gate = true>
__attribute__((always_inline)) inline bool llk_pack_reconfig_table_case(
    const std::uint32_t src, const std::uint32_t dst, const std::uint32_t faces) {
    constexpr ckernel::packer::llk_pack_reconfig_profiles_t prof =
        ckernel::packer::llk_pack_reconfig_profiles(pack_src_format, pack_dst_format, pack_tile_num_faces);
    if constexpr (k >= prof.n) {
        return false;
    } else {
        if (src == prof.src[k] && dst == prof.dst[k] && faces == prof.faces[k]) {
            constexpr ckernel::packer::llk_pack_reconfig_imm_t imm = ckernel::packer::
                llk_pack_reconfig_imm<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
                    pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
            ckernel::packer::reconfig_packer_data_format_imm_and_state<
                is_fp32_dest_acc_en,
                imm.strides_full,
                imm.format,
                imm.section,
                imm.lf8,
                imm.threshold,
                imm.dstacc,
                imm.ystride,
                imm.zwstride,
                prof.src[k],
                prof.dst[k],
                prof.faces[k]>();
            return true;
        }
        return llk_pack_reconfig_table_case<is_fp32_dest_acc_en, k + 1, gate>(src, dst, faces);
    }
}

// Out of line, as the reconfigure it replaces; an output of none of the profiles (an invalid id) keeps the full reconfigure.
template <bool is_fp32_dest_acc_en, bool gate = true>
__attribute__((noinline)) void llk_pack_reconfig_data_format_table(const std::uint32_t output_id) {
    if (!llk_pack_reconfig_table_case<is_fp32_dest_acc_en, 0, gate>(
            pack_src_format[output_id], pack_dst_format[output_id], pack_tile_num_faces[output_id])) {
        constexpr std::uint32_t skip = ckernel::packer::
            llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
                pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, skip>(
            pack_src_format[output_id],
            pack_dst_format[output_id],
            get_local_cb_interface(output_id).fifo_page_size,
            get_output_tile_c_dim(output_id),
            get_output_num_faces(output_id),
            false /* partial_face */);
    }
}

// The table above, inline for the one-tile wait form: a kernel without a one-tile block drops it with its branch.
template <bool is_fp32_dest_acc_en>
__attribute__((always_inline)) inline void llk_pack_reconfig_data_format_table_inl(const std::uint32_t output_id) {
    if (!llk_pack_reconfig_table_case<is_fp32_dest_acc_en, 0, false>(
            pack_src_format[output_id], pack_dst_format[output_id], pack_tile_num_faces[output_id])) {
        constexpr std::uint32_t skip = ckernel::packer::
            llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), false>(
                pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, skip, true>(
            pack_src_format[output_id],
            pack_dst_format[output_id],
            get_local_cb_interface(output_id).fifo_page_size,
            get_output_tile_c_dim(output_id),
            get_output_num_faces(output_id),
            false /* partial_face */);
    }
}

/**
 * Reconfigure the packer for a new output operand's data format.
 *
 * Face geometry (face_r_dim, num_faces), tile column dimension and tile size are derived from
 * the new output's CB metadata.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @param  new_output          New output circular buffer / operand index to configure for.
 */
// gate: the u2d reconfigure gate (llk_pack_reconfig_u2d_half); without it the pack waits for the unpacker instead.
// inl: no out-of-line copy (the one-tile wait form, which a kernel without a one-tile block drops with its branch).
template <bool is_fp32_dest_acc_en, bool gate, bool inl = false>
inline void llk_pack_reconfig_data_format_body(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::TileWidth>(tile_c_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateVal<Operand<Exu::Pack>::PartialFace>(false),
        StateDiscard<std::uint32_t>(get_local_cb_interface(output_id).fifo_page_size)));
    LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
    if constexpr (!gate) {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::UNPACK0);
    }
    [[maybe_unused]] constexpr std::uint32_t full_skip = ckernel::packer::
        llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
            pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
    // SDPA kernels: after a push of 8+ tiles (the k512 q chunk's QK row) the full reconfigure, whose GPR writes overlap the
    // packer drain; after a shorter one the immediates (joint_sdpa k512 bfp8 +0.7 % vs k128 gains).
    if constexpr (llk_pack_reconfig_table_ok<is_fp32_dest_acc_en, gate>() && ckernel::packer::llk_pack_sdpa_api<gate * 0 + 1>::value) {
        if (ckernel::packer::pack_sdpa_last_push >= 8) {
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, full_skip, inl, LLK_SIZE_MODE>(
                pack_src_format[output_id],
                pack_dst_format[output_id],
                get_local_cb_interface(output_id).fifo_page_size,
                tile_c_dim,
                num_faces,
                false /* partial_face */);
            return;
        }
    }
    if constexpr (llk_pack_reconfig_table_ok<is_fp32_dest_acc_en, gate>() && inl) {
        llk_pack_reconfig_data_format_table_inl<is_fp32_dest_acc_en>(output_id);
    } else if constexpr (llk_pack_reconfig_table_ok<is_fp32_dest_acc_en, gate>()) {
        llk_pack_reconfig_data_format_table<is_fp32_dest_acc_en, gate>(output_id);
    } else {
        constexpr std::uint32_t skip = ckernel::packer::
            llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
                pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, skip, inl, LLK_SIZE_MODE>(
            pack_src_format[output_id],
            pack_dst_format[output_id],
            get_local_cb_interface(output_id).fifo_page_size,
            tile_c_dim,
            num_faces,
            false /* partial_face */);
    }
}

// A constant-output reconfigure to the output whose configuration is live writes nothing (pack_live_output); not in size mode,
// nor at half sync where its writes keep the pack off the unpackers' L1 grants (the u2d gate, unpack-bound matmul). A run-time
// output keeps the full reconfigure: its compare, on every real switch, spills registers in the switching loop.
template <bool is_fp32_dest_acc_en>
constexpr bool llk_pack_live_skip() {
    constexpr bool skip_all =
        (ckernel::packer::llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), true>(
             pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face) &
         ckernel::packer::LLK_PACK_RECONFIG_SKIP_ALL) != 0;
    return !LLK_SIZE_MODE && !skip_all &&
           !ckernel::packer::llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, DST_SYNC_MODE>() &&
           !(DST_SYNC_MODE == DstSync::SyncHalf &&
             ckernel::packer::llk_pack_late_start<static_cast<std::uint32_t>(is_fp32_dest_acc_en)>(is_fp32_dest_acc_en));
}

// The kernel's reconfigures (reconfig_data_format.h marks, 256 + the output, 256 + 0xFF for a run-time one) all name outputs
// of one pack format: on a real switch between formats the skip's compare and store cost the loop more than it saves.
template <std::uint32_t dep, int... n>
constexpr bool llk_pack_live_one_format_of(std::integer_sequence<int, n...>) {
    constexpr bool m[] = {ckernel::packer::llk_pack_out_marked<256 + n, dep>(0)...};
    if (ckernel::packer::llk_pack_out_marked<256 + 0xFF, dep>(0)) {
        return false;
    }
    int f = -1;
    for (int i = 0; i < static_cast<int>(sizeof...(n)); i++) {
        if (m[i]) {
            if (f >= 0 && pack_dst_format[i] != f) {
                return false;
            }
            f = pack_dst_format[i];
        }
    }
    return true;
}
// The live-output byte of an output: 1 + the first output whose pack configuration (formats and tile geometry) it shares, so
// a reconfigure between outputs that differ only in their CB writes nothing.
template <std::uint32_t o>
constexpr std::uint8_t llk_pack_live_class() {
    for (std::uint32_t i = 0; i < o; i++) {
        if (pack_src_format[i] == pack_src_format[o] && pack_dst_format[i] == pack_dst_format[o] &&
            pack_tile_num_faces[i] == pack_tile_num_faces[o] && pack_partial_face[i] == pack_partial_face[o] &&
            pack_tile_face_r_dim[i] == pack_tile_face_r_dim[o] && pack_narrow_tile[i] == pack_narrow_tile[o] &&
            pack_tile_r_dim[i] == pack_tile_r_dim[o] && pack_tile_c_dim[i] == pack_tile_c_dim[o]) {
            return static_cast<std::uint8_t>(i + 1);
        }
    }
    return static_cast<std::uint8_t>(o + 1);
}

template <std::uint32_t dep>
constexpr bool llk_pack_live_one_format() {
    return llk_pack_live_one_format_of<dep>(std::make_integer_sequence<int, sizeof(pack_dst_format)>{});
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    if constexpr (ckernel::packer::llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, DST_SYNC_MODE>()) {
        // A 32-bit output keeps the gated full reconfigure, where a wait for the unpacker costs its longer packs; after
        // a one-tile block it waits instead.
        constexpr int kind = ckernel::packer::llk_pack_out32_kind(pack_dst_format);
        if constexpr (kind == 0) {
            llk_pack_reconfig_data_format_body<is_fp32_dest_acc_en, false>(new_output);
            return;
        } else if constexpr (kind == 2) {
            if (!ckernel::packer::llk_pack_out32(pack_dst_format[get_output_id(new_output)])) {
                llk_pack_reconfig_data_format_body<is_fp32_dest_acc_en, false>(new_output);
                return;
            }
        }
        if (ckernel::packer::pack_short_block) {
            llk_pack_reconfig_data_format_body<is_fp32_dest_acc_en, false, true>(new_output);
            return;
        }
    }
    llk_pack_reconfig_data_format_body<is_fp32_dest_acc_en, true>(new_output);
}

/**
 * llk_pack_reconfig_data_format for an output known at compile time (pack_reconfig_data_format's constant call sites):
 * where the kernel allows it (llk_pack_reconfig_imm), only the config bits its pack formats change, as immediates.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam new_output          New output circular buffer / operand index to configure for.
 */
template <bool is_fp32_dest_acc_en, std::uint32_t new_output, bool gate, bool inl = false>
__attribute__((always_inline)) inline void llk_pack_reconfig_data_format_ct_gate() {
    constexpr ckernel::packer::llk_pack_reconfig_imm_t imm =
        ckernel::packer::llk_pack_reconfig_imm<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
            pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face);
    // get_output_id is the identity.
    constexpr std::uint32_t output_id = new_output;
    // A SKIP_ALL kernel keeps the no-op reconfigure.
    constexpr bool skip_all =
        (ckernel::packer::llk_pack_reconfig_skip_all<is_fp32_dest_acc_en, DST_SYNC_MODE, sizeof(pack_dst_format), gate>(
             pack_src_format, pack_dst_format, pack_tile_num_faces, pack_partial_face) &
         ckernel::packer::LLK_PACK_RECONFIG_SKIP_ALL) != 0;
    if constexpr (
        !imm.ok || skip_all || output_id >= sizeof(pack_dst_format) ||
        pack_src_format[output_id] == static_cast<unsigned char>(DataFormat::Invalid) ||
        pack_dst_format[output_id] == static_cast<unsigned char>(DataFormat::Invalid)) {
        llk_pack_reconfig_data_format_body<is_fp32_dest_acc_en, gate, inl>(new_output);
    } else {
        SAN_HOOK(reconfigure(
            StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
            StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
            StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
            StateVal<Operand<Exu::Pack>::TileWidth>(get_output_tile_c_dim(output_id)),
            StateVal<Operand<Exu::Pack>::NumFaces>(get_output_num_faces(output_id)),
            StateVal<Operand<Exu::Pack>::PartialFace>(false),
            StateDiscard<std::uint32_t>(get_local_cb_interface(output_id).fifo_page_size)));
        LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en);
        if constexpr (!gate) {
            TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::UNPACK0);
        }
        ckernel::packer::reconfig_packer_data_format_imm_and_state<
            is_fp32_dest_acc_en,
            imm.strides_full,
            imm.format,
            imm.section,
            imm.lf8,
            imm.threshold,
            imm.dstacc,
            imm.ystride,
            imm.zwstride,
            pack_src_format[output_id],
            pack_dst_format[output_id],
            pack_tile_num_faces[output_id]>();
    }
}

template <bool is_fp32_dest_acc_en, std::uint32_t new_output>
__attribute__((always_inline)) inline void llk_pack_reconfig_data_format_ct() {
    if constexpr (!LLK_SIZE_MODE && ckernel::packer::llk_pack_sdpa_api<new_output * 0 + 1>::value) {
        if (ckernel::packer::pack_sdpa_last_push >= 8) {
            llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(new_output);
            return;
        }
    }
    if constexpr (LLK_SIZE_MODE) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(new_output);
    } else if constexpr (
        ckernel::packer::llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, DST_SYNC_MODE>() &&
        new_output < sizeof(pack_dst_format)) {
        if constexpr (ckernel::packer::llk_pack_out32(pack_dst_format[new_output])) {
            if (ckernel::packer::pack_short_block) {
                llk_pack_reconfig_data_format_ct_gate<is_fp32_dest_acc_en, new_output, false, true>();
            } else {
                llk_pack_reconfig_data_format_ct_gate<is_fp32_dest_acc_en, new_output, true>();
            }
        } else {
            llk_pack_reconfig_data_format_ct_gate<is_fp32_dest_acc_en, new_output, false>();
        }
    } else if constexpr (llk_pack_live_skip<is_fp32_dest_acc_en>() && llk_pack_live_one_format<new_output>()) {
        constexpr std::uint8_t live = llk_pack_live_class<new_output>();
        if (ckernel::packer::pack_live_output != live) {
            llk_pack_reconfig_data_format_ct_gate<is_fp32_dest_acc_en, new_output, true>();
            ckernel::packer::pack_live_output = live;
        }
    } else {
        llk_pack_reconfig_data_format_ct_gate<is_fp32_dest_acc_en, new_output, true>();
    }
}

/**
 * Conditionally reconfigure the packer when switching output operands: only reprograms the data
 * format when the new output's destination format differs from the old one and neither is Invalid.
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

    if ((pack_dst_format[old_output_id] != pack_dst_format[new_output_id]) &&
        (pack_dst_format[old_output_id] != (uint)DataFormat::Invalid) &&
        (pack_dst_format[new_output_id] != (uint)DataFormat::Invalid)) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(new_output);
    }
}

/**
 * llk_pack_reconfig_data_format(old_output, new_output) for a new output known at compile time: the same output-format
 * check, then llk_pack_reconfig_data_format_ct. The first form takes old_output at run time, the second at compile time.
 */
template <bool is_fp32_dest_acc_en, std::uint32_t new_output>
__attribute__((always_inline)) inline void llk_pack_reconfig_data_format_ct2(const std::uint32_t old_output) {
    if constexpr (new_output >= sizeof(pack_dst_format)) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(old_output, new_output);
    } else if constexpr (pack_dst_format[new_output] != static_cast<unsigned char>(DataFormat::Invalid)) {
        const std::uint32_t old_output_id = get_output_id(old_output);
        if ((pack_dst_format[old_output_id] != pack_dst_format[new_output]) &&
            (pack_dst_format[old_output_id] != (uint)DataFormat::Invalid)) {
            llk_pack_reconfig_data_format_ct<is_fp32_dest_acc_en, new_output>();
        }
    }
}

template <bool is_fp32_dest_acc_en, std::uint32_t old_output, std::uint32_t new_output>
__attribute__((always_inline)) inline void llk_pack_reconfig_data_format_ct2() {
    if constexpr (old_output >= sizeof(pack_dst_format) || new_output >= sizeof(pack_dst_format)) {
        llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(old_output, new_output);
    } else if constexpr (
        (pack_dst_format[old_output] != pack_dst_format[new_output]) &&
        (pack_dst_format[old_output] != static_cast<unsigned char>(DataFormat::Invalid)) &&
        (pack_dst_format[new_output] != static_cast<unsigned char>(DataFormat::Invalid))) {
        llk_pack_reconfig_data_format_ct<is_fp32_dest_acc_en, new_output>();
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
