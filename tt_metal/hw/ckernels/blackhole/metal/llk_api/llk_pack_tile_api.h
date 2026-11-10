// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_matmul_forms.h"
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

// The per-tile default pack writes the destination by bytes under full DEST sync, where the packer is the bound; under half sync the earlier pack
// start costs the unpackers L1 bank grants in latency-bound kernels. LLK_PACK_RMW_DEST forces it on everywhere, LLK_PACK_RMW_DEST_OFF off.
#if defined(LLK_PACK_RMW_DEST_OFF)
inline constexpr bool llk_pack_rmw_gate = false;
#else
inline constexpr bool llk_pack_rmw_gate = (DST_SYNC_MODE == DstSync::SyncFull);
#endif
template <PackMode pack_mode>
inline constexpr bool llk_pack_rmw_tile = llk_pack_rmw_dest || (llk_pack_rmw_gate && pack_mode == PackMode::Default);

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
    if constexpr (llk_pack_rmw_tile<pack_mode>) {
        _llk_pack_rmw_dest_init_<DST_SYNC_MODE == DstSync::SyncFull>();
    }
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_impl(std::uint32_t tile_index, std::uint32_t pack_tile_addr) {
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, llk_pack_rmw_tile<pack_mode>>(tile_index, pack_tile_addr);
}

// A Float32 output from a 16 bit DEST keeps main's destination write under full sync: the packer is long enough that
// the stall the byte-wise write saves hides nothing, and its two extra config stores cost 0.1 cycle per tile. The
// output is a compile-time constant in the kernels, so the test folds away.
template <bool is_fp32_dest_acc_en, PackMode pack_mode>
inline void llk_pack_tile_of_output(
    std::uint32_t tile_index, std::uint32_t pack_tile_addr, const std::uint8_t output_id) {
    if constexpr (llk_pack_rmw_tile<pack_mode> && !llk_pack_rmw_dest && !is_fp32_dest_acc_en) {
        if (pack_dst_format[output_id] == static_cast<std::uint32_t>(DataFormat::Float32)) {
            _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, false>(tile_index, pack_tile_addr);
            return;
        }
    }
    llk_pack_impl<is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
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

    llk_pack_tile_of_output<is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr, output_id);
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

        llk_pack_tile_of_output<is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr, output_id);
    }
}

// Matmul output blocks take llk_pack_block where the tt-llk sweep reads it faster: DestSync full (the pack is not hidden behind
// the math there, also at LoFi with several K tiles), and half sync at LoFi with one K tile; a static math throttle of 4 or more at half sync keeps the per-tile pack (math bounds the block).
// in_format, block_rt and block_ct are optional keys of the caller (Invalid and 0 keep the rule above): the operand format and the output block
// shape of the half sync cells where the block pack reads faster than the per-tile one; with in0_format given the keyed decision is matmul_pack_form_k (llk_matmul_forms.h).
template <
    MathFidelity math_fidelity,
    std::uint32_t kt_per_section,
    std::uint32_t output,
    int throttle_level = 0,
    DataFormat in_format = DataFormat::Invalid,
    std::uint32_t block_rt = 0,
    std::uint32_t block_ct = 0,
    DataFormat in0_format = DataFormat::Invalid>
constexpr bool llk_matmul_block_pack();
template <std::uint32_t output, std::uint32_t matmul_kt, std::uint32_t ntiles>
constexpr bool llk_matmul_pack_always_per_tile();
template <
    MathFidelity math_fidelity,
    std::uint32_t kt_per_section,
    std::uint32_t output,
    int throttle_level,
    DataFormat in_format,
    std::uint32_t block_rt,
    std::uint32_t block_ct,
    DataFormat in0_format>
constexpr bool llk_matmul_block_pack() {
    constexpr bool keyed = in_format != DataFormat::Invalid && block_rt != 0 && block_ct != 0;
    constexpr DataFormat out_format = static_cast<DataFormat>(pack_dst_format[output]);
    if constexpr (keyed && in0_format != DataFormat::Invalid) {
#ifdef PACKER_L1_ACC
        constexpr bool keyed_l1_acc = true;
#else
        constexpr bool keyed_l1_acc = false;
#endif
        return matmul_pack_form_k(
                   math_fidelity, DST_SYNC_MODE != DstSync::SyncFull, throttle_level, kt_per_section, block_rt, block_ct, in_format, in0_format, out_format, pack_tile_r_dim[output],
                   pack_tile_c_dim[output], false, keyed_l1_acc) != 0 &&
               !llk_matmul_pack_always_per_tile<output, kt_per_section, block_rt * block_ct>();
    } else if constexpr (
        keyed && DST_SYNC_MODE != DstSync::SyncFull && throttle_level == 0 &&
        (in_format == DataFormat::Float16_b || in_format == DataFormat::Float32) && kt_per_section == 1 && math_fidelity == MathFidelity::HiFi2 &&
        out_format != DataFormat::Float32 && out_format != DataFormat::Bfp8_b && out_format != DataFormat::Bfp4_b && pack_tile_num_faces[output] == 4 &&
        pack_tile_face_r_dim[output] == FACE_R_DIM && pack_partial_face[output] == 0 && block_rt * block_ct == 2) {
        return true;
    } else if constexpr (
        keyed && DST_SYNC_MODE != DstSync::SyncFull && throttle_level == 0 && in_format == DataFormat::Float32 &&
        math_fidelity == MathFidelity::HiFi2 && out_format == DataFormat::Bfp8_b && pack_tile_num_faces[output] == 4 &&
        pack_tile_face_r_dim[output] == FACE_R_DIM && pack_partial_face[output] == 0 && block_rt == 1 && block_ct >= 5) {
        return true;
    } else if constexpr (
        keyed && DST_SYNC_MODE != DstSync::SyncFull && throttle_level == 0 && in_format == DataFormat::Float16_b &&
        math_fidelity == MathFidelity::LoFi && kt_per_section == 1 && out_format == DataFormat::Float32 &&
        (pack_tile_num_faces[output] != 4 || pack_tile_face_r_dim[output] != FACE_R_DIM || pack_partial_face[output] != 0) && block_ct >= 4) {
        return true;
    } else if constexpr (
        keyed && DST_SYNC_MODE != DstSync::SyncFull && throttle_level == 0 && in_format == DataFormat::Bfp8_b && kt_per_section == 1 &&
        (math_fidelity == MathFidelity::LoFi || math_fidelity == MathFidelity::HiFi2) &&
        out_format == DataFormat::Float32 &&
        (pack_tile_num_faces[output] != 4 || pack_tile_face_r_dim[output] != FACE_R_DIM || pack_partial_face[output] != 0) && block_rt * block_ct >= 4) {
        return true;
    } else if constexpr (math_fidelity == MathFidelity::LoFi && kt_per_section > 1 && DST_SYNC_MODE != DstSync::SyncFull) {
        return false;
    } else if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        return true;
    } else if constexpr (throttle_level >= 4) {
        return false;
    } else {
        constexpr bool small_tile = pack_tile_num_faces[output] != 4 || pack_tile_face_r_dim[output] != FACE_R_DIM ||
                                    pack_partial_face[output] != 0;
        constexpr bool float32_out = out_format == DataFormat::Float32;
#ifdef PACKER_L1_ACC
        constexpr bool l1_acc = true;
#else
        constexpr bool l1_acc = false;
#endif
        // HiFi2 with one K tile: the per-tile pack is on the critical path for small tiles and Float32 outputs
        constexpr bool hifi2_one_k = math_fidelity == MathFidelity::HiFi2 && kt_per_section == 1 && (small_tile || float32_out);
        // The packer's L1 accumulate of a Float32 output reads and writes 64 byte rows; a block run of it reads slower than per-tile packs.
        return (math_fidelity == MathFidelity::LoFi || hifi2_one_k) && !(small_tile && float32_out) && !(l1_acc && float32_out);
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

// Which outputs pack 4 byte datums: two constant masks, like llk_pack_dst_bfp_mask.
constexpr std::uint32_t llk_pack_dst_wide_mask(const std::uint32_t first) {
    std::uint32_t mask = 0;
    for (std::uint32_t i = 0; i < 32; i++) {
        if (!IS_BFP_FORMAT(pack_dst_format[first + i]) && datum_size_in_bytes(pack_dst_format[first + i]) == 4) {
            mask |= 1u << i;
        }
    }
    return mask;
}
inline bool llk_pack_dst_is_wide(const std::uint32_t output_id) {
    constexpr std::uint32_t low = llk_pack_dst_wide_mask(0);
    constexpr std::uint32_t high = llk_pack_dst_wide_mask(32);
    return (((output_id < 32 ? low : high) >> (output_id & 31)) & 1) != 0;
}

// The largest page of a block pack, in 16 byte words: a 32x32 tile of 4 or 2 byte datums. Block floats and small tiles
// are smaller. Then 64 wide or 127 narrow tiles stay inside the 256 KB reach of the channel-1 W offset
// (_llk_pack_block_w_reach_), and a block of at most 16 tiles inside it too.
constexpr std::uint32_t LLK_PACK_PAGE_WORDS_WIDE = 256;
constexpr std::uint32_t LLK_PACK_PAGE_WORDS_NARROW = 128;
static_assert(16 * LLK_PACK_PAGE_WORDS_WIDE <= PACK_W_REACH_WORDS);
constexpr std::uint32_t LLK_PACK_W_MAX_TILES_WIDE = PACK_W_REACH_WORDS / LLK_PACK_PAGE_WORDS_WIDE;

// The CB page must fit the packer's W stride and the W offset reach.
inline bool llk_pack_block_is_closable(const std::uint32_t output_id) {
    return get_local_cb_interface(output_id).fifo_page_size <=
           (llk_pack_dst_is_wide(output_id) ? LLK_PACK_PAGE_WORDS_WIDE : LLK_PACK_PAGE_WORDS_NARROW);
}

#ifdef LLK_PACK_W_OFFSET
// W offset block packs: program the output base and the tile stride once per output and move the output of each block
// call with the packer W counter of channel 1. The base is programmed again when the output changes, the CB write
// pointer does not continue the last block (wrap, other packs into the CB), another pack wrote L1_Dest_addr or the
// channel-1 W stride in between (LLK_PACK_W_INVALIDATE), or the offset would pass LLK_PACK_W_MAX_TILES or the 256 KB
// reach of the W offset.
#ifndef LLK_PACK_W_MAX_TILES
#define LLK_PACK_W_MAX_TILES 127
#endif
static_assert(LLK_PACK_W_MAX_TILES * LLK_PACK_PAGE_WORDS_NARROW <= PACK_W_REACH_WORDS);
constexpr std::uint32_t LLK_PACK_W_MAX_TILES_W =
    LLK_PACK_W_MAX_TILES < LLK_PACK_W_MAX_TILES_WIDE ? LLK_PACK_W_MAX_TILES : LLK_PACK_W_MAX_TILES_WIDE;
// Returns whether this block takes the W offset path (out_tile is its tile offset). The first block of an output after another output packs as before: the base is
// programmed from the second block in a row on, so an output that alternates with another (partials and out of a spilled matmul) never pays the programming.
inline bool llk_pack_w_take(const std::uint8_t output_id, const std::uint32_t addr, const std::uint32_t page, const std::uint32_t ntiles, std::uint32_t& out_tile) {
    if (pack_w_last != output_id) {
        _llk_pack_close_fast_untilize_<DST_SYNC_MODE>();
        pack_w_last = output_id;
        return false;
    }
    const std::uint32_t max_tiles = llk_pack_dst_is_wide(output_id) ? LLK_PACK_W_MAX_TILES_W : LLK_PACK_W_MAX_TILES;
    const bool cont = pack_w_valid && pack_w_out == output_id && pack_w_page == page && pack_w_next == addr &&
                      pack_w_tiles + ntiles <= max_tiles;
    if (!cont) {
        _llk_pack_close_fast_untilize_<DST_SYNC_MODE>();
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

// Row of the form tables per output: tiles of 16x16, 16x32, 8x32, 4x32 and 1x32 (a face of 16, 8, 4 or fewer rows), Bfp8_b and Bfp8, Bfp4_b and Bfp2_b,
// Bfp4 and Bfp2, the other 4 byte datums, the other plain formats, Float16_b, Float32. A constant table, so the class of a call is one load.
constexpr std::uint8_t llk_pack_class_of(const std::uint32_t i) {
    const std::uint32_t rows = pack_tile_face_r_dim[i];
    const std::uint32_t fmt = pack_dst_format[i];
    if (pack_tile_num_faces[i] != 4 || rows != FACE_R_DIM) {
        if (rows == FACE_R_DIM) {
            return pack_tile_num_faces[i] == 1 ? 0 : 1;
        }
        return rows >= 8 ? 2 : (rows >= 4 ? 3 : 4);
    }
    if (fmt == static_cast<std::uint32_t>(DataFormat::Bfp4) || fmt == static_cast<std::uint32_t>(DataFormat::Bfp2)) {
        return 7;
    }
    if (IS_BFP_FORMAT(fmt)) {
        return fmt == static_cast<std::uint32_t>(DataFormat::Bfp8_b) || fmt == static_cast<std::uint32_t>(DataFormat::Bfp8) ? 5 : 6;
    }
    if (fmt == static_cast<std::uint32_t>(DataFormat::Float16_b)) {
        return 10;
    }
    if (fmt == static_cast<std::uint32_t>(DataFormat::Float32)) {
        return 11;
    }
    return datum_size_in_bytes(fmt) == 4 ? 8 : 9;
}
struct llk_pack_class_tab_t {
    std::uint8_t row[sizeof(pack_dst_format) / sizeof(pack_dst_format[0])];
    constexpr llk_pack_class_tab_t() : row() {
        for (std::uint32_t i = 0; i < sizeof(row); i++) {
            row[i] = llk_pack_class_of(i);
        }
    }
};
constexpr llk_pack_class_tab_t llk_pack_class_tab{};
inline std::uint32_t llk_pack_block_class(const std::uint32_t output_id) { return llk_pack_class_tab.row[output_id]; }

// Which outputs pack Int8 or UInt8: two constant masks. From a 16 bit DEST the per-tile pack writes two faces of them and the block forms four, so they stay per tile there.
constexpr std::uint32_t llk_pack_dst_int8_mask(const std::uint32_t first) {
    std::uint32_t mask = 0;
    for (std::uint32_t i = 0; i < 32; i++) {
        if (pack_dst_format[first + i] == static_cast<std::uint32_t>(DataFormat::Int8) ||
            pack_dst_format[first + i] == static_cast<std::uint32_t>(DataFormat::UInt8)) {
            mask |= 1u << i;
        }
    }
    return mask;
}
inline bool llk_pack_dst_is_int8(const std::uint32_t output_id) {
    constexpr std::uint32_t low = llk_pack_dst_int8_mask(0);
    constexpr std::uint32_t high = llk_pack_dst_int8_mask(32);
    return (((output_id < 32 ? low : high) >> (output_id & 31)) & 1) != 0;
}

// Which outputs, as inputs, are 32 bit (Float32, Int32, UInt32; the output CBs too, for their unpack_src_format is the output's): two constant masks.
constexpr std::uint32_t llk_pack_src32_mask(const std::uint32_t first) {
    std::uint32_t mask = 0;
    for (std::uint32_t i = 0; i < 32; i++) {
        if (unpack_src_format[first + i] == static_cast<std::uint32_t>(DataFormat::Float32) ||
            unpack_src_format[first + i] == static_cast<std::uint32_t>(DataFormat::Int32) ||
            unpack_src_format[first + i] == static_cast<std::uint32_t>(DataFormat::UInt32)) {
            mask |= 1u << i;
        }
    }
    return mask;
}
// A 32 bit CB next to a 32 bit DEST may unpack to DEST, where the unpacker bounds the loop and a block pack steals its L1 reads: those kernels take the forms no sweep read slower, the others the datacopy tables.
template <bool is_fp32_dest_acc_en>
inline constexpr bool llk_pack_src32_family = is_fp32_dest_acc_en && (llk_pack_src32_mask(0) != 0 || llk_pack_src32_mask(32) != 0);
// Whether a CB of the kernel unpacks to DEST (the JIT defines LLK_UNPACK_TO_DEST_ANY); without it one may. A kernel that unpacks to SrcA only takes the datacopy tables.
#if defined(LLK_UNPACK_TO_DEST_ANY)
inline constexpr bool llk_pack_unpack_to_dest = LLK_UNPACK_TO_DEST_ANY != 0;
#else
inline constexpr bool llk_pack_unpack_to_dest = true;
#endif

// Form of a block per tile count (above 16: the entry of 16): T per-tile packs, B the block that programs the output, W the block with the W offset, where no cell of the row read slower in the tt-llk sweeps.
// Columns: half sync 16 and 32 bit DEST, full sync 16 and 32 bit DEST; the first block of an output takes the init or the reconfigure table, the blocks that continue it the next table. The _dc tables hold the rows of the datacopy family.
constexpr char llk_pack_init_form[12][4][17] = {
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x16
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 8x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 4x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 1x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"},  // Bfp8_b, Bfp8
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"},  // Bfp4_b, Bfp2_b
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBTBTTTTTTTTTTT"},  // Bfp4, Bfp2
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // other 4 byte datums
    {"TBBBBBBBBBBBBBBB", "TBBTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // other plain formats
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Float16_b
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Float32
};
constexpr char llk_pack_reconf_form[12][4][17] = {
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x16
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 8x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 4x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 1x32
    {"TBBBBBBBBBBBBBBB", "TBTBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"},  // Bfp8_b, Bfp8
    {"TBBBBBBBBBBBBBBB", "TBBTTTTTTTTTTTTT", "TBBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"},  // Bfp4_b, Bfp2_b
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Bfp4, Bfp2
    {"BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // other 4 byte datums
    {"BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // other plain formats
    {"BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // Float16_b
    {"BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // Float32
};
constexpr char llk_pack_next_form[12][4][17] = {
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBWBBBBBBBB", "TWWBBBBBBBBBBBBB"},  // tile 16x16
    {"TWWWBWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBBBBBWWWWW", "TWWBBBBBBBBBBBBB"},  // tile 16x32
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"},  // tile 8x32
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"},  // tile 4x32
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBWWWWWWWWW"},  // tile 1x32
    {"TWBBBBWWWWWWWWWW", "TBWTTTTTTTTTTTTT", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"},  // Bfp8_b, Bfp8
    {"TWBBWWWWWWWWWWWW", "TBBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TBWWWWWWWWWWWWWW"},  // Bfp4_b, Bfp2_b
    {"TWBBWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"},  // Bfp4, Bfp2
    {"BWWWBBBBBBBBBBBB", "TBWBBBBBBBBBBBBB", "BWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // other 4 byte datums
    {"BBBWWWWWWWWWWWWW", "TBWWWWWWWWWWWWWW", "BBBWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // other plain formats
    {"WWWWWWWWWWWWWWWW", "TBBBBBBBBBBBBBBB", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // Float16_b
    {"WWWWBBBBBBBBBBBB", "TBWBBBBBBBBBBBBB", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // Float32
};
constexpr char llk_pack_init_form_dc[12][4][17] = {
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x16
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 8x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 4x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 1x32
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"},  // Bfp8_b, Bfp8
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"},  // Bfp4_b, Bfp2_b
    {"TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBTBTTTTTTTTTTT"},  // Bfp4, Bfp2
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // other 4 byte datums
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // other plain formats
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Float16_b
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Float32
};
constexpr char llk_pack_reconf_form_dc[12][4][17] = {
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x16
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 16x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 8x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 4x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // tile 1x32
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Bfp8_b, Bfp8
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Bfp4_b, Bfp2_b
    {"TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"},  // Bfp4, Bfp2
    {"BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // other 4 byte datums
    {"BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // other plain formats
    {"BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // Float16_b
    {"BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"},  // Float32
};
constexpr char llk_pack_next_form_dc[12][4][17] = {
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBWBBBBBBBB", "TWWBBBBBBBBBBBBB"},  // tile 16x16
    {"TWWWBWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBBBBBWWWWW", "TWWBBBBBBBBBBBBB"},  // tile 16x32
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"},  // tile 8x32
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"},  // tile 4x32
    {"TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBWWWWWWWWW"},  // tile 1x32
    {"TWBBBBWWWWWWWWWW", "TWBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"},  // Bfp8_b, Bfp8
    {"TWBBWWWWWWWWWWWW", "TWBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"},  // Bfp4_b, Bfp2_b
    {"TWBBWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"},  // Bfp4, Bfp2
    {"BWWWBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "BWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // other 4 byte datums
    {"BBBWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "BBBWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // other plain formats
    {"WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // Float16_b
    {"WWWWBBBBBBBBBBBB", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"},  // Float32
};

// Behind matmul math (a matmul init ran, or the K is given) a block can read slower than per tile in cells the sweeps measured, or a W offset block faster than the form of the table: bit (column * 16 + n - 1) of a mask forbids the W offset (no_w: the block instead), then the block (no_block) at n tiles,
// or, with the K given, takes the W offset (take_w) or the block (take_b). Rows: LoFi K not given, K 1, K above 1, HiFi2 to HiFi4 together, then LoFi and HiFi in a kernel
// without a matmul init (the other unpack and math contexts; take_w and take_b only with the K: without the first group or the last two); one entry per row of the form tables.
constexpr std::uint64_t llk_pack_mm_no_block[2][6][12] = {
    {  // datacopy family
        {0x0020002000000020, 0x00000000000800D8, 0, 0x0000000000000020, 0, 0x00000000000800A8, 0x0000000000080028, 0x00000000000000E8, 0x0000000000000080, 0x00000000000000A0, 0x00000000000000A0, 0x0000000000000080},  // LoFi, K not given
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // LoFi, K 1
        {0x0020002000000020, 0x00000000000800D8, 0, 0x0000000000000020, 0, 0x00000000000800A8, 0x0000000000080028, 0x00000000000000E8, 0x0000000000000080, 0x00000000000000A0, 0x00000000000000A0, 0x0000000000000080},  // LoFi, K above 1
        {0x00000000000E00FE, 0x00000000000A00AA, 0x00000000000A00EA, 0x00000000000E00CE, 0x00000000000E00EE, 0x0000000000020062, 0x0000000000020022, 0x0000000000000022, 0x0000000000000040, 0, 0, 0x0000000000000040},  // HiFi
        {0x0020002000020022, 0x00000000000A00DA, 0x0000000000020002, 0x0000000000020022, 0x0000000000020002, 0x00020002000A00AA, 0x00020002000A002A, 0x00160002000600EA, 0x0000000100000081, 0x00010001000100A1, 0x00010001000100A1, 0x0000000100000081},  // LoFi, no matmul init
        {0x00000000000E00FE, 0x00000000000A00AA, 0x00000000000A00EA, 0x00000000000E00CE, 0x00000000000E00EE, 0x0002000200020062, 0x0002000200020022, 0x0016000200060022, 0x0000000100000041, 0x0001000100010001, 0x0001000100010001, 0x0000000100000041},  // HiFi, no matmul init
    },
    {  // 32 bit input family
        {0x0020002000000020, 0x00000000000C00D8, 0, 0x0000000000040020, 0x00000000000C0000, 0x00000000000E00A8, 0x0000000000080028, 0x00000000000000E8, 0x0000000000000080, 0x00000000000000A0, 0x00000000000000A0, 0x0000000000000080},  // LoFi, K not given
        {0, 0, 0, 0, 0, 0x0000000000080000, 0, 0, 0, 0, 0, 0},  // LoFi, K 1
        {0x0020002000000020, 0x00000000000C00D8, 0, 0x0000000000040020, 0x00000000000C0000, 0x00000000000E00A8, 0x0000000000080028, 0x00000000000000E8, 0x0000000000000080, 0x00000000000000A0, 0x00000000000000A0, 0x0000000000000080},  // LoFi, K above 1
        {0x00000000000E00FE, 0x00000000000E00AA, 0x00000000000E00EA, 0x00000000000E00CE, 0x00000000000E00EE, 0x0000000000060062, 0x0000000000020022, 0x0000000000000022, 0x0000000000000040, 0, 0, 0x0000000000000040},  // HiFi
        {0x0020002000020020, 0x00000000000E00D8, 0x0000000000020000, 0x0000000000060020, 0x00000000000E0000, 0x00000000000E00A8, 0x00000000000A0028, 0x00160000000600E8, 0x0001000000010080, 0x00010000000100A0, 0x00010000000100A0, 0x0001000000010080},  // LoFi, no matmul init
        {0x00000000000E00FE, 0x00000000000E00AA, 0x00000000000E00EA, 0x00000000000E00CE, 0x00000000000E00EE, 0x0000000000060062, 0x0000000000020022, 0x0016000000060022, 0x0001000000010040, 0x0001000000010000, 0x0001000000010000, 0x0001000000010040},  // HiFi, no matmul init
    },
};
constexpr std::uint64_t llk_pack_mm_no_w[2][6][12] = {
    {  // datacopy family
        {0x000E000E00000020, 0x0000000000000090, 0, 0x0000000000000060, 0x0002000000000080, 0x0000000A000800B8, 0x0002000A00080028, 0x00000002000000E8, 0, 0x00000000000000A0, 0x00000000000000A0, 0},  // LoFi, K not given
        {0x0002000200000000, 0, 0, 0, 0x0002000000000000, 0, 0, 0, 0, 0, 0, 0},  // LoFi, K 1
        {0x000E000E00000020, 0x0000000000000090, 0, 0x0000000000000060, 0x0002000000000080, 0x0000000A000800B8, 0x0002000A00080028, 0x00000002000000E8, 0, 0x00000000000000A0, 0x00000000000000A0, 0},  // LoFi, K above 1
        {0x00020000000E00EE, 0x00000000000E00AE, 0x00000000000A00EA, 0x00000000000E00CE, 0x00000000000E00AE, 0x0000000000020062, 0x0000000000020022, 0x0000000000000022, 0x0000000000000040, 0x0000000000020002, 0x0000000000020002, 0x0000000000000040},  // HiFi
        {0x000E000E00000020, 0x0004000000000090, 0x0004004000000000, 0x0004000000000060, 0x0006000000000080, 0x0000000A000A00BA, 0x0002000A000A002A, 0x00000002000200EA, 0x0000000100000009, 0x00010001000100A1, 0x00010001000100A1, 0x0000000100000009},  // LoFi, no matmul init
        {0x00060000000E00EE, 0x00040000000E00AE, 0x00040040000A00EA, 0x00040000000E00CE, 0x00040000000E00AE, 0x0000000000020062, 0x0000000000020022, 0x0000000000020022, 0x0000000100000049, 0x0001000100030003, 0x0001000100030003, 0x0000000100000049},  // HiFi, no matmul init
    },
    {  // 32 bit input family
        {0x000E000E00000020, 0x00000000000C0090, 0x0000000000040000, 0x0000000000000060, 0x0002000000080080, 0x0000000A000A00B8, 0x0002000A000A0028, 0x00000002000000E8, 0x00000000000A0000, 0x00000000000200A0, 0x00000000000200A0, 0x00000000000A0000},  // LoFi, K not given
        {0x0002000200000000, 0, 0, 0, 0x0002000000000000, 0x00000000000A0000, 0x00020000000A0000, 0, 0, 0, 0, 0},  // LoFi, K 1
        {0x000E000E00000020, 0x00000000000C0090, 0x0000000000040000, 0x0000000000000060, 0x0002000000080080, 0x0000000A000A00B8, 0x0002000A000A0028, 0x00000002000000E8, 0x00000000000A0000, 0x00000000000200A0, 0x00000000000200A0, 0x00000000000A0000},  // LoFi, K above 1
        {0x00020000000E00EE, 0x00000000000E00AE, 0x00000000000E00EA, 0x00000000000E00CE, 0x00020000000E00AE, 0x00000000000E0062, 0x0002000000020022, 0x0000000000000022, 0x0000000000000040, 0x0000000000020002, 0x0000000000020002, 0x0000000000000040},  // HiFi
        {0x000E000E00000020, 0x00040000000C0090, 0x0004000000040000, 0x0004000000000060, 0x0006000000080080, 0x0000000A000A00B8, 0x0002000A000A0028, 0x00000002000200E8, 0x00010000000B0000, 0x00010000000300A0, 0x00010000000300A0, 0x00010000000B0000},  // LoFi, no matmul init
        {0x00060000000E00EE, 0x00040000000E00AE, 0x00040000000E00EA, 0x00040000000E00CE, 0x00060000000E00AE, 0x00000000000E0062, 0x0002000000020022, 0x0000000000020022, 0x0001000000010040, 0x0001000000030002, 0x0001000000030002, 0x0001000000010040},  // HiFi, no matmul init
    },
};
constexpr std::uint64_t llk_pack_mm_take_w[2][3][12] = {
    {  // datacopy family
        {0x00F8FF7800000000, 0x00F807F800000010, 0x00F8FF3800000000, 0x00F8FF3800000000, 0x0078FF3800000000, 0x00000000000C003C, 0x00000000000C000C, 0x000000000000000C, 0, 0, 0, 0x00000000000000F0},  // LoFi, K 1
        {0x0020D02000000000, 0x00F807F800000000, 0x00F8FF3800000000, 0x00F8FF3800000000, 0x0078FF3800000000, 0x0000000000040004, 0, 0x0000000000000004, 0, 0, 0, 0x00000000000000F0},  // LoFi, K above 1
        {0x00F8FF7800000000, 0x00F807F800000000, 0x00F8FF3800000000, 0x00F8FF3800000000, 0x0078FF3800000000, 0x00000000000C001C, 0, 0, 0, 0, 0, 0x00000000000000B0},  // HiFi
    },
    {  // 32 bit input family
        {0x00F8FF7800000000, 0x00F807F800000010, 0x00F8FF3800000000, 0x00F8FF3800000000, 0x0078FF3800000000, 0x000000000000003C, 0x000200000006000C, 0x000000000000000C, 0, 0, 0x00000000000E0000, 0x00000000000A00F0},  // LoFi, K 1
        {0x00E0D02000000000, 0x00F807F800000000, 0x00F8FF3800000000, 0x00F8FF3800000000, 0x0078FF3800000000, 0x0000000000020004, 0x0000000000060000, 0x0000000000000004, 0, 0, 0x00000000000C0000, 0x00000000000000F0},  // LoFi, K above 1
        {0x00F8FF7800000000, 0x00F807F800000000, 0x00F8FF3800000000, 0x00F8FF3800000000, 0x0078FF3800000000, 0x000000000000001C, 0x0002000000000000, 0, 0, 0, 0x0000000000040000, 0x00000000000A00B0},  // HiFi
    },
};
constexpr std::uint64_t llk_pack_mm_take_b[2][3][12] = {
    {  // datacopy family
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // LoFi, K 1
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // LoFi, K above 1
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // HiFi
    },
    {  // 32 bit input family
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // LoFi, K 1
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // LoFi, K above 1
        {0, 0, 0, 0, 0, 0x0000000000080000, 0, 0, 0, 0, 0, 0},  // HiFi
    },
};

// Set by the matmul inits on the pack thread: the masks are for blocks behind matmul math. Only matmul kernels write it, so the others fold the test away.
[[maybe_unused]] static bool llk_pack_matmul_seen = false;
inline void llk_pack_matmul_context() { llk_pack_matmul_seen = true; }

template <bool is_fp32_dest_acc_en, std::uint32_t matmul_kt>
inline char llk_pack_block_form(const std::uint32_t output_id, const std::uint32_t ntiles, const char (*table)[4][17], const bool next) {
    const std::uint32_t row = llk_pack_block_class(output_id);
    const std::uint32_t column = (DST_SYNC_MODE == DstSync::SyncFull ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);
    const std::uint32_t count = (ntiles < 16 ? ntiles : 16) - 1;
    if (!is_fp32_dest_acc_en && llk_pack_dst_is_int8(output_id)) {
        return 'T';
    }
    const char form = table[row][column][count];
    constexpr std::uint32_t family = llk_pack_src32_family<is_fp32_dest_acc_en> ? 1 : 0;
    constexpr std::uint32_t group = MATH_FIDELITY != MathFidelity::LoFi ? 3 : (matmul_kt == 0 ? 0 : (matmul_kt == 1 ? 1 : 2));
    constexpr std::uint32_t group_other = MATH_FIDELITY != MathFidelity::LoFi ? 5 : 4;
    const std::uint32_t mask_group = (matmul_kt != 0 || llk_pack_matmul_seen) ? group : group_other;
    const std::uint32_t bit = column * 16 + count;
    char out = form;
    if (out == 'W' && ((llk_pack_mm_no_w[family][mask_group][row] >> bit) & 1) != 0) {
        out = 'B';
    }
    if (out == 'B' && ((llk_pack_mm_no_block[family][mask_group][row] >> bit) & 1) != 0) {
        out = 'T';
    }
    if constexpr (matmul_kt != 0) {
        if (next && out != 'W' && ((llk_pack_mm_take_w[family][group - 1][row] >> bit) & 1) != 0) {
            out = 'W';
        }
        if (next && out == 'T' && ((llk_pack_mm_take_b[family][group - 1][row] >> bit) & 1) != 0) {
            out = 'B';
        }
    }
    return out;
}

template <bool is_fp32_dest_acc_en>
inline const char (*llk_pack_form_table(const char (*u)[4][17], const char (*dc)[4][17]))[4][17] {
    return llk_pack_src32_family<is_fp32_dest_acc_en> && llk_pack_unpack_to_dest ? u : dc;
}

// The per-tile packs of a block that takes none of the block forms: main's loop.
template <bool is_fp32_dest_acc_en>
inline void llk_pack_block_per_tile(const std::uint32_t start_tile_index, const std::uint8_t output_id, const std::uint32_t ntiles) {
    _llk_pack_close_fast_untilize_<DST_SYNC_MODE>();
    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr = get_output_tile_address<false, PackMode::Default>(output_id, 0);
        llk_pack_tile_of_output<is_fp32_dest_acc_en, PackMode::Default>(tile_index, pack_tile_addr, output_id);
    }
}

// guard_dest_straddle: two block-float tiles at DEST tiles 3 and 4 read 6 cycles slower closed than per tile, so a caller with a variable start tile takes the per-tile path.
// Off by default, the check costs instructions in the callers that pass 0.
// matmul_kt: the K tiles per output block of the matmul that filled DEST, where the caller knows it at compile time; 0 takes the forms no K reads slower than per tile.
template <bool is_fp32_dest_acc_en, bool guard_dest_straddle = false, std::uint32_t matmul_kt = 0>
__attribute__((always_inline)) inline void llk_pack_block(std::uint32_t start_tile_index, std::uint32_t output, std::uint32_t ntiles) {
    std::uint8_t output_id = get_output_id(output);
#ifndef LLK_PACK_W_OFFSET
    _llk_pack_close_fast_untilize_<DST_SYNC_MODE>();
#endif

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

    // The block forms run the default pack program: after a tilize, untilize or pack rows init, main's per-tile packs. So too while the packer
    // accumulates in L1 under half sync, where a block behind matmul math reads up to 64 cycles slower (full sync: faster).
    if (LLK_PACK_OTHER_PROGRAM_ACTIVE() || (DST_SYNC_MODE == DstSync::SyncHalf && LLK_PACK_L1_ACC_ACTIVE())) {
        llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles);
        return;
    }

#ifdef LLK_PACK_W_OFFSET
    const bool first_block = pack_w_last != output_id;
    const char form = llk_pack_block_form<is_fp32_dest_acc_en, matmul_kt>(
        output_id,
        ntiles,
        first_block ? (pack_reconfigured ? llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_reconf_form, llk_pack_reconf_form_dc)
                                         : llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_init_form, llk_pack_init_form_dc))
                    : llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_next_form, llk_pack_next_form_dc),
        !first_block);
    pack_reconfigured = false;
    pack_w_last = output_id;
#else
    const char form = llk_pack_block_form<is_fp32_dest_acc_en, matmul_kt>(
        output_id, ntiles, llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_init_form, llk_pack_init_form_dc), false);
#endif
    if (form == 'T') {
        llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles);
        return;
    }
    const bool take_w = form == 'W';
    if (!take_w) {
        _llk_pack_close_fast_untilize_<DST_SYNC_MODE>();
    }
    if (llk_pack_block_is_contiguous(output_id)) {
        std::uint32_t pack_tile_addr =
            get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
        get_local_cb_interface(output_id).fifo_wr_tile_ptr +=
            get_local_cb_interface(output_id).fifo_page_size * ntiles;
#ifdef LLK_PACK_W_OFFSET
        std::uint32_t out_tile = 0;
        if (take_w && llk_pack_w_take(output_id, pack_tile_addr, get_local_cb_interface(output_id).fifo_page_size, ntiles, out_tile)) {
            _llk_pack_block_w_<DST_SYNC_MODE, is_fp32_dest_acc_en>(start_tile_index, out_tile, ntiles);
        } else {
            _llk_pack_block_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, llk_pack_rmw_tile<PackMode::Default>>(start_tile_index, pack_tile_addr, ntiles);
        }
#else
        _llk_pack_block_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default, llk_pack_rmw_tile<PackMode::Default>>(
            start_tile_index, pack_tile_addr, ntiles);
#endif
    } else if (
        llk_pack_block_is_closable(output_id) &&
        !(guard_dest_straddle && IS_BFP_FORMAT(pack_dst_format[output_id]) && ntiles == 2 && (start_tile_index & 3) == 3)) {
        const std::uint32_t page_size = get_local_cb_interface(output_id).fifo_page_size;
        std::uint32_t pack_tile_addr =
            get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
        get_local_cb_interface(output_id).fifo_wr_tile_ptr += page_size * ntiles;
#ifdef LLK_PACK_W_OFFSET
        std::uint32_t out_tile = 0;
        if (take_w && llk_pack_w_take(output_id, pack_tile_addr, page_size, ntiles, out_tile)) {
            _llk_pack_block_closed_w_mop_<DST_SYNC_MODE, is_fp32_dest_acc_en>(start_tile_index, out_tile, ntiles, get_output_num_faces(output_id));
        } else {
            _llk_pack_block_closed_<DST_SYNC_MODE, is_fp32_dest_acc_en, llk_pack_rmw_tile<PackMode::Default>>(
                start_tile_index, pack_tile_addr, ntiles, page_size << 4, get_output_num_faces(output_id));
        }
#else
        _llk_pack_block_closed_<DST_SYNC_MODE, is_fp32_dest_acc_en, llk_pack_rmw_tile<PackMode::Default>>(
            start_tile_index, pack_tile_addr, ntiles, page_size << 4, get_output_num_faces(output_id));
#endif
    } else {
        llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles);
    }
}

// Whether the prepared block takes the closed form, whose destination words llk_pack_block_prepare can issue early. Int8 and UInt8 from a 16 bit DEST stay per tile, as in llk_pack_block.
template <bool is_fp32_dest_acc_en>
inline bool llk_pack_block_is_closed_form(std::uint32_t output, std::uint32_t ntiles) {
    const std::uint8_t output_id = get_output_id(output);
    return ntiles > 1 && (is_fp32_dest_acc_en || !llk_pack_dst_is_int8(output_id)) && !llk_pack_block_is_contiguous(output_id) &&
           llk_pack_block_is_closable(output_id);
}

// Full sync only: issues the closed block pack's destination words before the math wait of the block.
template <bool is_fp32_dest_acc_en, std::uint32_t matmul_kt = 0>
inline void llk_pack_block_prepare(std::uint32_t start_tile_index, std::uint32_t output, std::uint32_t ntiles) {
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        if (llk_pack_block_is_closed_form<is_fp32_dest_acc_en>(output, ntiles)) {
            const std::uint8_t output_id = get_output_id(output);
            _llk_pack_close_fast_untilize_<DST_SYNC_MODE>();
            _llk_pack_block_closed_pre_<llk_pack_rmw_tile<PackMode::Default>>(
                start_tile_index,
                get_local_cb_interface(output_id).fifo_wr_ptr + get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1);
        }
    }
}

// llk_pack_block after llk_pack_block_prepare of the same block, with no pack in between.
template <bool is_fp32_dest_acc_en, std::uint32_t matmul_kt = 0>
inline void llk_pack_block_prepared(std::uint32_t start_tile_index, std::uint32_t output, std::uint32_t ntiles) {
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        if (llk_pack_block_is_closed_form<is_fp32_dest_acc_en>(output, ntiles)) {
            const std::uint8_t output_id = get_output_id(output);
            const std::uint32_t page_size = get_local_cb_interface(output_id).fifo_page_size;
            get_local_cb_interface(output_id).fifo_wr_tile_ptr += page_size * ntiles;
            _llk_pack_block_closed_run_(ntiles, page_size << 4, get_output_num_faces(output_id));
            return;
        }
    }
    llk_pack_block<is_fp32_dest_acc_en, false, matmul_kt>(start_tile_index, output, ntiles);
}

// BEGIN GENERATED by tt-llk/tests/python_tests/gen_matmul_pack_tables.py: llk_pack_block_form and llk_pack_block_class of plain values; do not edit
template <bool is_fp32_dest_acc_en, std::uint32_t matmul_kt>
constexpr char llk_pack_block_form_k(const std::uint32_t row, const std::uint32_t ntiles, const char (*table)[4][17], const bool next, const bool sync_full, const bool src32, const bool int8) {
    const std::uint32_t column = (sync_full ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);
    const std::uint32_t count = (ntiles < 16 ? ntiles : 16) - 1;
    if (!is_fp32_dest_acc_en && int8) {
        return 'T';
    }
    const char form = table[row][column][count];
    const std::uint32_t family = (is_fp32_dest_acc_en && src32) ? 1 : 0;
    constexpr std::uint32_t group = MATH_FIDELITY != MathFidelity::LoFi ? 3 : (matmul_kt == 0 ? 0 : (matmul_kt == 1 ? 1 : 2));
    const std::uint32_t bit = column * 16 + count;
    char out = form;
    if (out == 'W' && ((llk_pack_mm_no_w[family][group][row] >> bit) & 1) != 0) {
        out = 'B';
    }
    if (out == 'B' && ((llk_pack_mm_no_block[family][group][row] >> bit) & 1) != 0) {
        out = 'T';
    }
    if constexpr (matmul_kt != 0) {
        if (next && out != 'W' && ((llk_pack_mm_take_w[family][group - 1][row] >> bit) & 1) != 0) {
            out = 'W';
        }
        if (next && out == 'T' && ((llk_pack_mm_take_b[family][group - 1][row] >> bit) & 1) != 0) {
            out = 'B';
        }
    }
    return out;
}

constexpr std::uint32_t llk_pack_block_class_k(const std::uint32_t rows, const std::uint32_t faces, const bool bfpa, const bool bfp, const bool bfp8, const bool wide) {
    if (faces != 4 || rows != FACE_R_DIM) {
        if (rows == FACE_R_DIM) {
            return faces == 1 ? 0 : 1;
        }
        return rows >= 8 ? 2 : (rows >= 4 ? 3 : 4);
    }
    if (bfpa) {
        return 7;
    }
    if (bfp) {
        return bfp8 ? 5 : 6;
    }
    return wide ? 8 : 9;
}
// END GENERATED

// Whether llk_pack_block<DST_ACCUM_MODE, false, matmul_kt> takes the per-tile form for every block of a matmul output of ntiles tiles (the first, after a
// reconfigure, the next ones): llk_matmul_block_pack keeps the per-tile pack there.
template <std::uint32_t output, std::uint32_t matmul_kt, std::uint32_t ntiles>
constexpr bool llk_matmul_pack_always_per_tile() {
    constexpr std::uint32_t f = pack_dst_format[output];
    constexpr bool bfp = IS_BFP_FORMAT(f);
    constexpr std::uint32_t row = llk_pack_block_class_k(
        pack_tile_face_r_dim[output],
        pack_tile_num_faces[output],
        f == static_cast<std::uint32_t>(DataFormat::Bfp4) || f == static_cast<std::uint32_t>(DataFormat::Bfp2),
        bfp,
        f == static_cast<std::uint32_t>(DataFormat::Bfp8_b) || f == static_cast<std::uint32_t>(DataFormat::Bfp8),
        !bfp && datum_size_in_bytes(f) == 4);
    constexpr bool int8 = f == static_cast<std::uint32_t>(DataFormat::Int8) || f == static_cast<std::uint32_t>(DataFormat::UInt8);
    constexpr bool src32 = llk_pack_src32_family<DST_ACCUM_MODE>;
    constexpr bool t32 = src32 && llk_pack_unpack_to_dest;
    constexpr bool full = DST_SYNC_MODE == DstSync::SyncFull;
    return llk_pack_block_form_k<DST_ACCUM_MODE, matmul_kt>(row, ntiles, t32 ? llk_pack_init_form : llk_pack_init_form_dc, false, full, src32, int8) == 'T' &&
           llk_pack_block_form_k<DST_ACCUM_MODE, matmul_kt>(row, ntiles, t32 ? llk_pack_reconf_form : llk_pack_reconf_form_dc, false, full, src32, int8) == 'T' &&
           llk_pack_block_form_k<DST_ACCUM_MODE, matmul_kt>(row, ntiles, t32 ? llk_pack_next_form : llk_pack_next_form_dc, true, full, src32, int8) == 'T';
}
