// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL implementation of the unpacker hardware configuration (Blackhole).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

using namespace ckernel;

namespace hal_llk
{

// Reset the unpacker Y/Z/W address counters. The X counters (datum range) are owned by the op init.
inline void unpacker_reset_address_counters()
{
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .apply();
}

// Select configuration context 0 for both unpackers and restart the software context rotation.
inline void unpacker_reset_config_context()
{
    unpack_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

// Program the per-unpacker tile descriptor, output format and channel-1 (register side) strides.
template <cfg::Sec S>
inline void unpacker_configure_one(const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    const std::uint32_t face_dim  = face_r_dim * FACE_C_DIM;
    const std::uint32_t x_stride  = register_datum_bytes(dst_format);
    const std::uint32_t y_stride  = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride  = FACE_R_DIM * y_stride;
    constexpr std::uint32_t U     = (S == cfg::Sec::S0) ? 0 : 1;

    // L1-side tile geometry: one face is one Z plane of face_dim datums.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, S>(format_code(src_format)),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, S>(face_dim),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, S>(num_faces),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.WDim, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsYStartLo, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsYStartHi, S, 0>());

    // Register-side output format and unpacker mode (no tilize, no upsample, uncompressed contexts).
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, S>(format_code(dst_format)),
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
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx2, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx3, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count_non_log2_en, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx4, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx5, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx6, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx7, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_limit_address, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_fifo_size, S, 0>());

    if constexpr (U == 0)
    {
        // Unpacker 0 takes the face width from its per-context X dimension (the UNPACR overrides the
        // thread context), so the descriptor X dimension stays zero.
        cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, S, 0>();
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, S>(face_dim), cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, S>(face_dim));
    }
    else
    {
        // Unpacker 1 takes the face width from its tile descriptor; its per-context fields stay zero.
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, S, 0>(),
            cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, S, 0>(),
            cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, S, 0>(),
            cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, S, 0>());
    }

    // Channel-1 (SrcA/SrcB) strides: one face row (SrcA only) and one face.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(U == 0 ? y_stride : 0),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
}

} // namespace hal_llk

/**
 * @brief Configure the unpacker hardware for both operands A and B.
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
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // No tile of this kernel may still own an unpacker context, and the unpackers must be idle
    // before their configuration changes.
    while (sync::semaphore::read<hal_llk::UNPACK_SYNC_SEM>() > 0)
    {
    }
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    hal_llk::unpacker_reset_address_counters();

    hal_llk::unpacker_configure_one<cfg::Sec::S0>(unpA_src_format, unpA_dst_format, unpA_face_r_dim, unpA_num_faces);
    hal_llk::unpacker_configure_one<cfg::Sec::S1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim, unpB_num_faces);

    // SrcA formats / accumulation mode seen by the matrix unit and SFPU.
    const bool int8_math = hal_llk::is_int8_math_format(unpA_dst_format) || hal_llk::is_int8_math_format(unpB_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(hal_llk::is_uint8_format(unpA_dst_format) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(hal_llk::is_uint8_format(unpB_dst_format) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));

    // Unpacker 0 writes SrcA face by face: the per-context destination address is combined with the
    // channel-1 counter, and the SrcA set follows the write address.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, hal_llk::UNPACK_SRCA_DEST_ADDRESS>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, hal_llk::UNPACK_SRCA_DEST_ADDRESS>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].ForcedSharedExp, cfg::Sec::S0, 0>(), cfg::set<cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());

    hal_llk::unpacker_reset_config_context();
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg                 = hal::cfg;
    constexpr std::uint32_t fpu   = (stoch_rnd_mode == StochRndType::Fpu || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    constexpr std::uint32_t pack  = (stoch_rnd_mode == StochRndType::Pack || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack>());
}
