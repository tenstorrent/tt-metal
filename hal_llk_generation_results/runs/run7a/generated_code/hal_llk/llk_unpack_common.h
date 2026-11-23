// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole unpack common LLK functions.

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk::unpack
{

using AdcClient  = hal::AddressCounterClient;
using AdcChannel = hal::AddressChannel;

// Program the tile descriptor and the unpacker configuration words of one unpacker.
// Every field of the owned words is written so that no state from an earlier kernel leaks in.
template <cfg::Sec S>
inline void configure_unpacker(
    const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t tile_x_dim, const std::uint32_t num_faces)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        // Tile descriptor (input tile geometry in L1)
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, S>(src_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, S>(tile_x_dim),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, S>(num_faces),
        // Unpacker configuration (output format, throttling, no tilize/upsample/shift, uncompressed contexts)
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, S>(dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, S, 2>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Haloize_mode, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Tileize_mode, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_Src_Reg_Set_Upd, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_If_Sel, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_rate, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Ovrd_data_format, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_and_interleave, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx0, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx1, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx2, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx3, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx2, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx3, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count_non_log2, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count_non_log2_en, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx4, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx5, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx6, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx7, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Metadata_x_end, S, 0>());
}

} // namespace hal_llk::unpack

/**
 * @brief Configure both unpackers (formats, tile geometry, output strides) and the ALU source formats.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_num_faces,
    const std::uint32_t unpB_num_faces,
    [[maybe_unused]] const std::uint32_t unpA_tile_size = 0,
    [[maybe_unused]] const std::uint32_t unpB_tile_size = 0)
{
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;
    using hal_llk::fmt;
    using hal_llk::unpack::AdcChannel;
    using hal_llk::unpack::AdcClient;

    // Both unpackers must be idle before their configuration is replaced.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // Reset all unpacker address counters except the channel-1 X (datum count) counters.
    hal::address_counters.client<AdcClient::Unpacker0, AdcClient::Unpacker1>()
        .channel<AdcChannel::Channel0>()
        .X<0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .channel<AdcChannel::Channel1>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .apply();

    // ALU source formats and Dest accumulation mode (also programmed by the math thread).
    const std::uint32_t int8_math_enabled = hal_llk::is_int8_math_format(unpA_dst_format) || hal_llk::is_int8_math_format(unpB_dst_format);
    sync::mutex::acquire<sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::Dstacc_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::Dstacc_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>((unpA_dst_format & 0x1F) == fmt(DataFormat::UInt8)),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>((unpB_dst_format & 0x1F) == fmt(DataFormat::UInt8)),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(unpA_dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(unpB_dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math_enabled));
    sync::mutex::release<sync::Mutex::M0>();

    // FP8 E4M3 input exponent mode is shared by both unpackers.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0>(
        (unpA_src_format & 0x1F) == fmt(DataFormat::Fp8_e4m3));

    // Tile descriptors and unpacker configuration. SrcA X dim comes from the per-context Tile_x_dim below.
    hal_llk::unpack::configure_unpacker<cfg::Sec::S0>(unpA_src_format, unpA_dst_format, 0, unpA_num_faces);
    hal_llk::unpack::configure_unpacker<cfg::Sec::S1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim * FACE_C_DIM, unpB_num_faces);

    // Channel-1 (register side) strides: one face row per Y step, one face per Z step.
    const std::uint32_t unpA_x_stride = hal_llk::register_datum_bytes(unpA_dst_format);
    const std::uint32_t unpB_x_stride = hal_llk::register_datum_bytes(unpB_dst_format);
    const std::uint32_t face_a_dim    = unpA_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(FACE_C_DIM * unpA_x_stride),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_C_DIM * FACE_R_DIM * unpA_x_stride),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_C_DIM * FACE_R_DIM * unpB_x_stride),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>(),
        // SrcA write base for both contexts, and per-context face size in L1
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, 4 * 16>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, 4 * 16>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0>(face_a_dim),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0>(face_a_dim),
        // Final SrcA address = context dest address + channel-1 counter
        cfg::set<cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>());

    // Use configuration context 0 only.
    unp_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>();
}

/**
 * @brief Select stochastic rounding for the FPU, packer gasket and packer.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg                 = hal_llk::cfg;
    constexpr std::uint32_t fpu   = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr std::uint32_t pack  = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Pack);
    hal_llk::sync::mutex::acquire<hal_llk::sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack>());
    hal_llk::sync::mutex::release<hal_llk::sync::Mutex::M0>();
}
