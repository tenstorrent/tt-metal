// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

/**
 * @brief No-op pack-side ordering primitive; present only for API parity with Quasar.
 *
 * WH/BH have no pack-side WAIT/PUSH ordering requirement, so this has no functional role in a kernel. It
 * exists only so the shared compute-API dummy_pack() resolves on every architecture; on Quasar the same
 * call is a required TEN-4746 drain primitive (see that arch's llk_pack_tile_api.h).
 *
 * @param pack_output  The output dataflow buffer identifier (unused).
 */
inline void llk_pack_dummy([[maybe_unused]] const std::uint32_t pack_output) {}

// Opt-in byte-wise destination write of the default packs (_llk_pack_rmw_dest_init_): a kernel defines LLK_PACK_RMW_DEST. Off, the packs write
// the destination as main does.
#ifdef LLK_PACK_RMW_DEST
inline constexpr bool llk_pack_rmw_dest = true;
#else
inline constexpr bool llk_pack_rmw_dest = false;
#endif

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/). They take
// already-resolved scalar formats/geometry + the runtime write address; the per-source prologue
// (resolving these from a CB id, or from an MemDescriptor) lives in the callers.
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init_impl(
    const std::uint32_t pack_src_reg_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool is_input_8bit_format) {
    // Blackhole's MOP does not take the format at all; set_packer_strides is its only consumer here and
    // reconfig_packer_data_format re-derives that, so a format reconfig between init and execute is legal
    // and must not read as drift.
    SAN_HOOK(init<OperationPack>(
        StateDiscard<std::uint32_t>(pack_src_reg_format),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::TileWidth>(tile_c_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateDiscard<bool>(is_input_8bit_format),
        StateVal<OperationPack::NumTiles>(num_tiles)));

    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_reg_format, face_r_dim, tile_c_dim, num_faces, num_tiles, is_input_8bit_format);
    if constexpr (llk_pack_rmw_dest) {
        _llk_pack_rmw_dest_init_();
    }
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_impl(std::uint32_t tile_index, std::uint32_t pack_tile_addr) {
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, llk_pack_rmw_dest>(tile_index, pack_tile_addr);
}

template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(
    const std::uint32_t pack_output, const std::uint32_t num_tiles, const std::uint32_t input_operand) {
    // TODO (https://github.com/tenstorrent/tt-metal/issues/18948): Revisit for narrow_tile
    const std::uint32_t output_id = get_output_id(pack_output);

    if constexpr (!skip_addrmod_config) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }

    // For pack with tilize enabled, check if the original input format is 8-bit.
    // 8-bit datums (Int8, UInt8, Fp8_e4m3, Lf8) do not require the tilize workaround on Blackhole.
    bool is_input_8bit_format = false;
    if constexpr (pack_mode == PackMode::Tilize) {
        is_input_8bit_format = IS_8BIT_FORMAT(static_cast<std::uint32_t>(unpack_src_format[input_operand]));
    }
    llk_pack_init_impl<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_format[output_id],
        get_output_face_r_dim(output_id),
        get_output_tile_c_dim(output_id),
        get_output_num_faces(output_id),
        num_tiles,
        is_input_8bit_format);
}

// input_operand is only consumed by the Blackhole tilize workaround. Keep the common non-tilize API
// architecture-agnostic without giving the input CB id a default value.
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t pack_output, const std::uint32_t num_tiles = 1) {
    static_assert(pack_mode != PackMode::Tilize, "PackMode::Tilize requires an explicit input_operand");
    llk_pack_init<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_output, num_tiles, 0 /* input_operand unused outside PackMode::Tilize */);
}

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr =
        get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    SAN_HOOK(execute<OperationPack>(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(get_output_face_r_dim(output_id)),
        StateVal<Operand<Exu::Pack>::TileWidth>(get_output_tile_c_dim(output_id)),
        StateVal<Operand<Exu::Pack>::NumFaces>(get_output_num_faces(output_id)),
        StateDiscard<std::uint32_t>(tile_index),
        StateDiscard<std::uint32_t>(output_tile_index)));

    llk_pack_impl<is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
}

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_matmul_pack(
    std::uint32_t start_tile_index, std::uint32_t output, std::uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");
    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    LLK_ASSERT(
        ((start_tile_index + ntiles - 1) < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    SAN_HOOK(execute<OperationPack>(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(get_output_face_r_dim(output_id)),
        StateVal<Operand<Exu::Pack>::TileWidth>(get_output_tile_c_dim(output_id)),
        StateVal<Operand<Exu::Pack>::NumFaces>(get_output_num_faces(output_id)),
        StateDiscard<std::uint32_t>(start_tile_index),
        StateDiscard<std::uint32_t>(ntiles),
        StateDiscard<std::uint32_t>(output_tile_index)));

    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr =
            get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, llk_pack_rmw_dest>(tile_index, pack_tile_addr);
    }
}

// Matmul output blocks take llk_pack_block where the tt-llk sweep reads it faster: DestSync full, and half sync at LoFi
// with one K tile; a static math throttle of 4 or more at half sync keeps the per-tile pack (math bounds the block).
template <MathFidelity math_fidelity, std::uint32_t kt_per_section, std::uint32_t output, int throttle_level = 0>
constexpr bool llk_matmul_block_pack() {
    if constexpr (math_fidelity == MathFidelity::LoFi && kt_per_section > 1) {
        return false;
    } else if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        return true;
    } else if constexpr (throttle_level >= 4) {
        return false;
    } else {
        constexpr bool small_tile = pack_tile_num_faces[output] != 4 || pack_tile_face_r_dim[output] != FACE_R_DIM ||
                                    pack_partial_face[output] != 0;
        constexpr bool float32_out = static_cast<DataFormat>(pack_dst_format[output]) == DataFormat::Float32;
#ifdef PACKER_L1_ACC
        constexpr bool l1_acc = true;
#else
        constexpr bool l1_acc = false;
#endif
        // The packer's L1 accumulate of a Float32 output reads and writes 64 byte rows; a block run of it reads slower than per-tile packs.
        return math_fidelity == MathFidelity::LoFi && !(small_tile && float32_out) && !(l1_acc && float32_out);
    }
}

// What one pack run can write back to back: plain format, full 32x32 tiles, one tile per CB page.
inline bool llk_pack_block_is_contiguous(const std::uint32_t output_id) {
    const std::uint32_t dst_format = pack_dst_format[output_id];
    if (IS_BFP_FORMAT(dst_format)) {
        return false;
    }
    if (get_output_num_faces(output_id) != 4 || get_output_face_r_dim(output_id) != FACE_R_DIM ||
        get_output_partial_face(output_id)) {
        return false;
    }
    const std::uint32_t tile_words = (TILE_R_DIM * TILE_C_DIM * datum_size_in_bytes(dst_format)) >> 4;
    return get_local_cb_interface(output_id).fifo_page_size == tile_words;
}

// The CB page must fit the packer's W stride.
inline bool llk_pack_block_is_closable(const std::uint32_t output_id) {
    return get_local_cb_interface(output_id).fifo_page_size < 0x1000;
}

#ifdef LLK_PACK_W_OFFSET
// W offset block packs: program the output base and the tile stride once per output and move the output of each block call with the packer W counter of channel 1.
// The base is programmed again when the output changes, the CB write pointer does not continue the last block (wrap, other packs into the CB), another pack wrote
// L1_Dest_addr or the channel-1 W stride in between (LLK_PACK_W_INVALIDATE), or the offset would pass LLK_PACK_W_MAX_TILES.
#ifndef LLK_PACK_W_MAX_TILES
#define LLK_PACK_W_MAX_TILES 127
#endif
// Returns whether this block takes the W offset path (out_tile is its tile offset). The first block of an output after another output packs as before: the base is
// programmed from the second block in a row on, so an output that alternates with another (partials and out of a spilled matmul) never pays the programming.
inline bool llk_pack_w_take(const std::uint8_t output_id, const std::uint32_t addr, const std::uint32_t page, const std::uint32_t ntiles, std::uint32_t& out_tile) {
    if (pack_w_last != output_id) {
        pack_w_last = output_id;
        return false;
    }
    const bool cont = pack_w_valid && pack_w_out == output_id && pack_w_page == page && pack_w_next == addr && pack_w_tiles + ntiles <= LLK_PACK_W_MAX_TILES;
    if (!cont) {
        program_packer_destination(addr);
        const std::uint32_t stride = page << 4;
        if (stride != pack_ch1_w_stride) {
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(stride);
            pack_ch1_w_stride = stride;
        }
        pack_w_valid = true;
        pack_w_out   = output_id;
        pack_w_page  = page;
        pack_w_next  = addr;
        pack_w_tiles = 0;
    }
    out_tile = pack_w_tiles;
    pack_w_tiles += ntiles;
    pack_w_next += page * ntiles;
    return true;
}
#endif

// Small tiles: one to four faces of 1 to 16 rows, at most eight PACRs, full tile width.
inline bool llk_pack_block_is_small(const std::uint32_t output_id) {
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    return get_output_tile_c_dim(output_id) == TILE_C_DIM && num_faces * (face_r_dim < 4 ? 1 : face_r_dim >> 2) <= 8;
}

// Small tiles of a plain format, one tile per CB page: they run back to back with one close.
inline bool llk_pack_block_is_dense(const std::uint32_t output_id) {
    const std::uint32_t dst_format = pack_dst_format[output_id];
    if (IS_BFP_FORMAT(dst_format) || !llk_pack_block_is_small(output_id)) {
        return false;
    }
    const std::uint32_t tile_words =
        (get_output_num_faces(output_id) * get_output_face_r_dim(output_id) * FACE_C_DIM * datum_size_in_bytes(dst_format)) >> 4;
    return get_local_cb_interface(output_id).fifo_page_size == tile_words;
}

// Records the tile body that llk_pack_block<is_fp32_dest_acc_en, true> replays for the small tiles of this output; call it after the pack init
// and again after any program that replaces the replay buffer.
inline void llk_pack_block_dense_init(const std::uint32_t output) {
    const std::uint8_t output_id = get_output_id(output);
    if (llk_pack_block_is_dense(output_id)) {
        _llk_pack_block_dense_init_(get_output_face_r_dim(output_id), get_output_num_faces(output_id));
    }
}

// Next to HiFi math the closed block-float pack makes the op read slower than per-tile packs, except for 4 and 8 tiles at HiFi4; HiFi2 never gained.
template <MathFidelity math_fidelity>
constexpr bool llk_pack_block_closed_pays(const bool block_float_out, const std::uint32_t ntiles) {
    return math_fidelity == MathFidelity::LoFi || !block_float_out ||
           (math_fidelity == MathFidelity::HiFi4 && (ntiles == 4 || ntiles == 8));
}

// Which outputs pack block float: two constant masks, so the per-call test is a shift, not a load and a format switch.
constexpr std::uint32_t llk_pack_dst_bfp_mask(const std::uint32_t first) {
    std::uint32_t mask = 0;
    for (std::uint32_t i = 0; i < 32; i++) {
        if (IS_BFP_FORMAT(pack_dst_format[first + i])) {
            mask |= 1u << i;
        }
    }
    return mask;
}
inline bool llk_pack_dst_is_bfp(const std::uint32_t output_id) {
    constexpr std::uint32_t low = llk_pack_dst_bfp_mask(0);
    constexpr std::uint32_t high = llk_pack_dst_bfp_mask(32);
    return (((output_id < 32 ? low : high) >> (output_id & 31)) & 1) != 0;
}

#ifndef LLK_PACK_BLOCK_MIN_TILES
#define LLK_PACK_BLOCK_MIN_TILES 4
#endif

// The per-tile packs of a block that takes none of the block forms: main's loop.
template <bool is_fp32_dest_acc_en>
inline void llk_pack_block_per_tile(const std::uint32_t start_tile_index, const std::uint8_t output_id, const std::uint32_t ntiles) {
    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr = get_output_tile_address<false, PackMode::Default>(output_id, 0);
        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, false, llk_pack_rmw_dest>(tile_index, pack_tile_addr);
    }
}

// dense_small: the caller called llk_pack_block_dense_init for this output, small tiles of a plain format then run back to back with one close.
// guard_dest_straddle: for a caller whose start tile index is not a constant 0: two block-float tiles that straddle a four tile boundary of DEST (tiles 3 and 4)
// read 6 cycles slower as a closed block than packed one by one, so they take the per-tile path. Off by default, the check costs instructions in the callers that pass 0.
template <bool is_fp32_dest_acc_en, bool dense_small = false, bool guard_dest_straddle = false>
inline void llk_pack_block(std::uint32_t start_tile_index, std::uint32_t output, std::uint32_t ntiles) {
    std::uint8_t output_id = get_output_id(output);
    // Full tiles in blocks of fewer tiles read slower than the per-tile pack in the tt-llk sweep; tiny tiles read faster as a closed block from two tiles on.
    const bool block_form = ntiles >= LLK_PACK_BLOCK_MIN_TILES || (ntiles >= 2 && llk_pack_block_is_small(output_id));

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    LLK_ASSERT(
        ((start_tile_index + ntiles - 1) < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    SAN_HOOK(execute<OperationPack>(
        StateVal<Operand<Exu::Pack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Pack>::InputFormat>(pack_src_format[output_id]),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(get_output_face_r_dim(output_id)),
        StateVal<Operand<Exu::Pack>::TileWidth>(get_output_tile_c_dim(output_id)),
        StateVal<Operand<Exu::Pack>::NumFaces>(get_output_num_faces(output_id)),
        StateDiscard<std::uint32_t>(start_tile_index),
        StateDiscard<std::uint32_t>(ntiles)));

    // Block-float outputs that keep the per-tile pack (HiFi, see llk_pack_block_closed_pays) skip the layout checks of the block forms.
    if (!block_form || (llk_pack_dst_is_bfp(output_id) && !llk_pack_block_closed_pays<MATH_FIDELITY>(true, ntiles))) {
        llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles);
        return;
    }
    if (llk_pack_block_is_contiguous(output_id)) {
        std::uint32_t pack_tile_addr =
            get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
        get_local_cb_interface(output_id).fifo_wr_tile_ptr +=
            get_local_cb_interface(output_id).fifo_page_size * ntiles;
#ifdef LLK_PACK_W_OFFSET
        std::uint32_t out_tile = 0;
        if (llk_pack_w_take(output_id, pack_tile_addr, get_local_cb_interface(output_id).fifo_page_size, ntiles, out_tile)) {
            _llk_pack_block_w_<DST_SYNC_MODE, is_fp32_dest_acc_en>(start_tile_index, out_tile, ntiles);
        } else {
            _llk_pack_block_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, llk_pack_rmw_dest>(start_tile_index, pack_tile_addr, ntiles);
        }
#else
        _llk_pack_block_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, llk_pack_rmw_dest>(
            start_tile_index, pack_tile_addr, ntiles);
#endif
    } else if (dense_small && llk_pack_block_is_dense(output_id)) {
        std::uint32_t pack_tile_addr =
            get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
        get_local_cb_interface(output_id).fifo_wr_tile_ptr +=
            get_local_cb_interface(output_id).fifo_page_size * ntiles;
        _llk_pack_block_dense_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
            start_tile_index, pack_tile_addr, ntiles, get_output_face_r_dim(output_id), get_output_num_faces(output_id));
    } else if (
        llk_pack_block_is_closable(output_id) &&
        !(guard_dest_straddle && IS_BFP_FORMAT(pack_dst_format[output_id]) && ntiles == 2 && (start_tile_index & 3) == 3)) {
        const std::uint32_t page_size = get_local_cb_interface(output_id).fifo_page_size;
        std::uint32_t pack_tile_addr =
            get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
        get_local_cb_interface(output_id).fifo_wr_tile_ptr += page_size * ntiles;
#ifdef LLK_PACK_W_OFFSET
        std::uint32_t out_tile = 0;
        if (llk_pack_w_take(output_id, pack_tile_addr, page_size, ntiles, out_tile)) {
            _llk_pack_block_closed_w_mop_<DST_SYNC_MODE, is_fp32_dest_acc_en>(start_tile_index, out_tile, ntiles, get_output_num_faces(output_id));
        } else {
            _llk_pack_block_closed_<DST_SYNC_MODE, is_fp32_dest_acc_en, llk_pack_rmw_dest>(
                start_tile_index, pack_tile_addr, ntiles, page_size << 4, get_output_num_faces(output_id));
        }
#else
        _llk_pack_block_closed_<DST_SYNC_MODE, is_fp32_dest_acc_en, llk_pack_rmw_dest>(
            start_tile_index, pack_tile_addr, ntiles, page_size << 4, get_output_num_faces(output_id));
#endif
    } else {
        llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles);
    }
}
