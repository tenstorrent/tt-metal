// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The per-tile pack of a CB output, llk_pack_block's fallback; the includer provides DST_SYNC_MODE and pack_dst_format.

#include <cstdint>

#include "llk_pack.h"

// Some CB of the kernel unpacks to DEST (LLK_UNPACK_TO_DEST_ANY of the JIT or the test shim); without it, assume one does.
#if defined(LLK_UNPACK_TO_DEST_ANY)
inline constexpr bool llk_pack_u2d = LLK_UNPACK_TO_DEST_ANY != 0;
#else
inline constexpr bool llk_pack_u2d = true;
#endif

namespace ckernel::packer
{
template <>
inline constexpr bool pack_reconfig_u2d<DstSync::SyncHalf> = llk_pack_u2d;
template <>
inline constexpr bool pack_strides_u2d<PackMode::Default> = llk_pack_u2d;
} // namespace ckernel::packer

// The block forms write the destination by bytes under full DEST sync only: under half sync the earlier pack start costs
// the unpackers L1 bank grants in latency-bound kernels.
template <PackMode pack_mode>
inline constexpr bool llk_pack_rmw_tile = DST_SYNC_MODE == DstSync::SyncFull && pack_mode == PackMode::Default;

// A 16-bit DEST matmul kernel of four CBs with a block-float in0 (CB 0), a Float16_b or Float32 in1 (CB 1) and a Float16_b output (CB 2):
// minimal 222 1x1 bfp8 x bf16 HiFi2, whose unpacker binds, and the byte-wise write's earlier start costs it L1 (+0.4..+0.8 %, e41d7fe1da8
// bisect). Block-float in1 or output keeps the byte-wise write (bmm Bfp8 1x1x1 +6 cy/block with the word write).
template <std::uint32_t dep>
constexpr bool llk_pack_matmul_bfp16()
{
    constexpr std::uint8_t inv = static_cast<std::uint8_t>(DataFormat::Invalid);
    if (sizeof(unpack_src_format) < 4)
    {
        return false;
    }
    bool four = true;
    for (std::uint32_t i = 0; i < sizeof(unpack_src_format); i++)
    {
        four = four && ((unpack_src_format[i] != inv) == (i < 4));
    }
    const std::uint8_t in1 = unpack_src_format[1];
    return four && IS_BFP_FORMAT(unpack_src_format[0]) &&
           (in1 == static_cast<std::uint8_t>(DataFormat::Float16_b) || in1 == static_cast<std::uint8_t>(DataFormat::Float32)) &&
           unpack_src_format[2] == static_cast<std::uint8_t>(DataFormat::Float16_b) && ckernel::packer::llk_pack_matmul_kernel<dep>() &&
           !ckernel::packer::llk_pack_sdpa_api<dep * 0 + 1>::value;
}

// The default per-tile pack writes the destination by bytes under both syncs, except at half sync where its earlier start
// contends with the unpackers for L1: fp32 DEST with an unpack to DEST, unpack-bound matmul (llk_pack_late_start), a 16-bit
// DEST matmul with a block-float CB, and a copy kernel (llk_pack_copy).
template <bool is_fp32_dest_acc_en, PackMode pack_mode>
inline constexpr bool llk_pack_rmw_per_tile =
    pack_mode == PackMode::Default &&
    (DST_SYNC_MODE == DstSync::SyncFull ||
     !((is_fp32_dest_acc_en && llk_pack_u2d) || ckernel::packer::llk_pack_late_start<static_cast<std::uint32_t>(pack_mode)>(is_fp32_dest_acc_en) ||
       (!is_fp32_dest_acc_en && llk_pack_matmul_bfp16<static_cast<std::uint32_t>(pack_mode)>()) ||
       ckernel::packer::llk_pack_copy<static_cast<std::uint32_t>(pack_mode)>(is_fp32_dest_acc_en)));

template <std::uint32_t dep>
constexpr DstSync llk_pack_sync_of()
{
    return DST_SYNC_MODE;
}

namespace ckernel::packer
{
template <>
struct llk_pack_pre_format_api<1>
{
    static constexpr bool of(const int cb)
    {
        return llk_pack_pre_format(pack_dst_format[cb]);
    }
};
} // namespace ckernel::packer

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/). They take
// already-resolved scalar formats/geometry + the runtime write address; the per-source prologue
// (resolving these from a CB id, or from an MemDescriptor) lives in the callers.
template <
    PackMode pack_mode       = PackMode::Default,
    bool zero_output         = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false,
    std::uint32_t live       = 0>
inline void llk_pack_init_impl(
    const std::uint32_t pack_src_reg_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool is_input_8bit_format)
{
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

    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, false, live>(
        pack_src_reg_format, face_r_dim, tile_c_dim, num_faces, num_tiles, is_input_8bit_format);
}

// llk_pack_init_live for this kernel: CBs of different widths keep the full init at half sync with fp32 DEST and an unpack to
// DEST, the contention class. dep: a template argument of the caller.
template <auto dep, bool is_fp32_dest_acc_en, DstSync sync = DST_SYNC_MODE>
constexpr bool llk_pack_init_live_kernel()
{
    return ckernel::packer::llk_pack_init_live<dep, sync>(pack_src_format, pack_dst_format) &&
           (ckernel::packer::llk_pack_width_single(pack_src_format, pack_dst_format) ||
            !(is_fp32_dest_acc_en && sync == DstSync::SyncHalf && ckernel::packer::pack_reconfig_u2d<sync>));
}

// The compute API's default init of an output (metal llk_pack_init, the tt-llk pack API shim). counted: the call passed a
// tile count, so it may write a MOP other than hw_configure's one-tile MOP (pack_mop_dirty); a kernel whose inits pass none
// never writes the flag, which then folds.
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool counted, bool is_fp32_dest_acc_en>
inline void llk_pack_init_of_output(const std::uint32_t output_id, const std::uint32_t num_tiles, const bool is_input_8bit_format)
{
    if constexpr (zero_output || pack_mode != PackMode::Default)
    {
        ckernel::packer::pack_mop_dirty = true;
    }
    else if constexpr (counted)
    {
        if (num_tiles != 1)
        {
            ckernel::packer::pack_mop_dirty = true;
        }
    }
    if constexpr (pack_mode == PackMode::Default && llk_pack_init_live_kernel<pack_mode, is_fp32_dest_acc_en>())
    {
        if (!LLK_PACK_OTHER_PROGRAM_ACTIVE())
        {
            constexpr std::uint32_t live =
                1 | (ckernel::packer::llk_pack_mop_live<pack_mode, DST_SYNC_MODE>(pack_src_format, pack_dst_format, pack_tile_face_r_dim, pack_tile_num_faces) ? 2 : 0) |
                (ckernel::packer::llk_pack_width_single(pack_src_format, pack_dst_format) ? 0 : 4);
            llk_pack_init_impl<pack_mode, zero_output, true, skip_packer_strides, live>(
                pack_src_format[output_id], pack_tile_face_r_dim[output_id], pack_tile_c_dim[output_id], pack_tile_num_faces[output_id], num_tiles, is_input_8bit_format);
            return;
        }
    }
    llk_pack_init_impl<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_format[output_id], pack_tile_face_r_dim[output_id], pack_tile_c_dim[output_id], pack_tile_num_faces[output_id], num_tiles, is_input_8bit_format);
}

// After the compute API's hw_configure: where the kernel's default inits keep its program, its address modes, X counter and,
// with llk_pack_mop_live, one-tile MOP.
template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure_live(const std::uint32_t face_r_dim, const std::uint32_t tile_c_dim, const std::uint32_t num_faces)
{
    if constexpr (llk_pack_init_live_kernel<is_fp32_dest_acc_en, is_fp32_dest_acc_en>())
    {
        if (!LLK_PACK_OTHER_PROGRAM_ACTIVE())
        {
            _llk_pack_configure_addrmod_<PackMode::Default>();
            if constexpr (ckernel::packer::llk_pack_mop_live<is_fp32_dest_acc_en, DST_SYNC_MODE>(
                              pack_src_format, pack_dst_format, pack_tile_face_r_dim, pack_tile_num_faces))
            {
                _llk_pack_mop_config_<PackMode::Default, false>(face_r_dim, tile_c_dim, num_faces, 1);
            }
            TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
        }
    }
}

// B3 (llk_pack_pre_dst_write) is on for this kernel's per-tile packs.
template <bool is_fp32_dest_acc_en, PackMode pack_mode>
constexpr bool llk_pack_pre_on()
{
    return !is_fp32_dest_acc_en && llk_pack_rmw_per_tile<is_fp32_dest_acc_en, pack_mode> &&
           ckernel::packer::llk_pack_pre_dst_sync<llk_pack_sync_of<static_cast<std::uint32_t>(pack_mode)>()>();
}

// A block pack that writes its destination itself sets the next prediction: the tile after its last, or 0.
template <bool is_fp32_dest_acc_en>
inline void llk_pack_pre_set(const std::uint32_t next)
{
    if constexpr (llk_pack_pre_on<is_fp32_dest_acc_en, PackMode::Default>())
    {
        ckernel::packer::pack_pre_next = next;
    }
}

// size: a kernel bound by the kernel config buffer (LLK_SIZE_MODE) takes main's destination write.
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool w_invalidate = true, bool size = false>
inline void llk_pack_impl(std::uint32_t tile_index, std::uint32_t pack_tile_addr)
{
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, llk_pack_rmw_per_tile<is_fp32_dest_acc_en, pack_mode> && !size, w_invalidate>(
        tile_index, pack_tile_addr);
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode, bool w_invalidate = true, bool size = false>
inline void llk_pack_tile_of_output(std::uint32_t tile_index, std::uint32_t pack_tile_addr, [[maybe_unused]] const std::uint8_t output_id)
{
    // The kernel's one output (llk_pack_one_output): its next tile follows this one, and a half-sync release writes it ahead
    // (llk_pack_pre_dst_write).
    if constexpr (llk_pack_pre_on<is_fp32_dest_acc_en, pack_mode>() && !size)
    {
        const std::uint32_t pre        = ckernel::packer::pack_pre_next;
        ckernel::packer::pack_pre_next = pack_tile_addr + get_local_cb_interface(output_id).fifo_page_size;
        if (tile_index == 0 && pre == (pack_tile_addr | ckernel::packer::LLK_PACK_PRE_WRITTEN))
        {
            ckernel::ckernel_template::run();
            return;
        }
    }
    llk_pack_impl<is_fp32_dest_acc_en, pack_mode, w_invalidate, size>(tile_index, pack_tile_addr);
}
