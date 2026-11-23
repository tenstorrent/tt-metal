// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the unpacker hardware configuration (spec: llk_lib/llk_unpack_common.h).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk::unpack
{

namespace cfg = hal::cfg;

// SrcA position (in datums) written by unpacker 0 for the first datum of a face (Thcon Reg5 Dest_cntx*_address).
// Observed on hardware: math-side SrcA row r (FPU reads, MOVD2A writes) is unpacker SrcA row r + 4,
// so faces are written from row 4.
constexpr std::uint32_t SRC_WRITE_BASE_DATUMS = 4 * FACE_C_DIM;

inline void wait_unpackers_idle_for_config()
{
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::Unpacker0Idle | hal::sync::StallCondition::Unpacker1Idle>();
}

// Program the per-unpacker tile descriptor, output format and SrcA/SrcB channel-1 strides.
template <std::uint32_t U>
inline void configure_unpacker(
    const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces, const bool transpose_xy)
{
    constexpr cfg::Sec S         = (U == 0) ? cfg::Sec::S0 : cfg::Sec::S1;
    const std::uint32_t x_stride = datum_bytes(dst_format);
    const bool lf8_4b_exp        = (src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);

    // Tile descriptor: uncompressed tile, one face per Z plane.
    // Unpacker 0 takes its X dim from the per-context Tile_x_dim registers (programmed below).
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, S>(format_code(src_format)),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, S>(U == 0 ? 0 : face_r_dim * FACE_C_DIM),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, S>(num_faces),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.WDim, S, 0>());

    // Unpacker configuration: output format, no tilize/upsample, uncompressed contexts.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, S>(format_code(dst_format)),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, S, 2>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Haloize_mode, S>(transpose_xy ? 1 : 0),
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
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_limit_address, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_fifo_size, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, S>(lf8_4b_exp ? 1 : 0));

    // Channel-1 (register side) strides: one face row and one face per Y/Z step.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(FACE_C_DIM * x_stride),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_R_DIM * FACE_C_DIM * x_stride),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
}

// Program the unpacker 0 face size (X dim per context) and the SrcA write base.
inline void configure_srca_face_dim(const std::uint32_t face_r_dim)
{
    const std::uint32_t face_dim = face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, SRC_WRITE_BASE_DATUMS>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, SRC_WRITE_BASE_DATUMS>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0>(face_dim),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0>(face_dim));
}

// Observed on hardware: in the SETADCXX form emitted for "X on both channels", the Channel0 value
// is the last datum index and the Channel1 value the first (see the HAL report).
template <std::uint32_t FaceRows>
inline void set_unpacker_x_range()
{
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X<FaceRows * FACE_C_DIM - 1>()
        .template channel<hal::AddressChannel::Channel1>()
        .template X<0>()
        .apply();
}

// Program the number of datums one UNPACR reads (one face): X runs from 0 to face_dim - 1.
inline void configure_unpacker_x_end(const std::uint32_t face_r_dim)
{
    switch (face_r_dim)
    {
        case 1:
            set_unpacker_x_range<1>();
            break;
        case 2:
            set_unpacker_x_range<2>();
            break;
        case 4:
            set_unpacker_x_range<4>();
            break;
        case 8:
            set_unpacker_x_range<8>();
            break;
        default:
            set_unpacker_x_range<FACE_R_DIM>();
            break;
    }
}

// Reset the Z/W (face/tile) counters of both unpackers on both channels.
inline void reset_unpacker_zw_counters()
{
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .W<0>()
        .apply();
}

} // namespace hal_llk::unpack

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
    namespace cfg = hal::cfg;
    using namespace hal_llk;

    unpack::wait_unpackers_idle_for_config();

    // Start from clean address counters on both unpackers.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X<0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .X<0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .apply();

    unpack::configure_unpacker<0>(unpA_src_format, unpA_dst_format, unpA_face_r_dim, unpA_num_faces, false /*transpose_xy*/);
    unpack::configure_unpacker<1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim, unpB_num_faces, false /*transpose_xy*/);
    unpack::configure_srca_face_dim(unpA_face_r_dim);

    // ALU formats follow the unpacker output formats; Dest accumulates in FP32 when requested.
    const bool int8_math = is_int8_or_int32(unpA_dst_format) || is_int8_or_int32(unpB_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(unpA_src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(unpB_src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(format_code(unpA_dst_format)),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(format_code(unpB_dst_format)),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));

    // Unpack thread state: explicit configuration context 0 without offsets, implied Src formats enabled.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, 0>();

    unp_cfg_context = 0;
}

template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg                = hal::cfg;
    constexpr std::uint32_t mode = static_cast<std::uint32_t>(stoch_rnd_mode);
    constexpr std::uint32_t fpu  = (mode & static_cast<std::uint32_t>(StochRndType::Fpu)) ? 1 : 0;
    constexpr std::uint32_t pack = (mode & static_cast<std::uint32_t>(StochRndType::Pack)) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack>());
}
