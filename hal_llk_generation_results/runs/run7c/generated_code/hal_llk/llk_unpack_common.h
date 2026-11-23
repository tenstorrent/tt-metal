// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the unpack-side common LLK functions (Blackhole).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_hal_common.h"

namespace hal_llk::unpack_detail
{

// Unpacker-0 per-context SrcA write base (THCON Dest_cntx*_address).
constexpr std::uint32_t SRCA_CONTEXT_ADDRESS = 4 * 16;

// Reset every X/Y/Z/W address counter of both unpackers, both channels.
inline void reset_address_counters()
{
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
}

// Program the unpacker datum count for one face: X runs from 0 to face_r_dim * 16 - 1.
// The HAL encodes the channel-0 X value into the SETADCXX end-of-row field and the channel-1 X value into the
// start field (observed with a compile-time encoding comparison), so the end goes on channel 0.
inline void set_face_datum_count(const std::uint32_t face_r_dim)
{
    const std::uint32_t x_end = (face_r_dim == 0) ? 1 : (face_r_dim * FACE_C_DIM) - 1;
    hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X(x_end)
        .channel<hal::AddressChannel::Channel1>()
        .X(0)
        .apply();
}

} // namespace hal_llk::unpack_detail

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
    using namespace hal_llk;

    // Nothing may still be reading the configuration we are about to replace.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    unpack_detail::reset_address_counters();

    const std::uint32_t a_x_stride = register_datum_bytes(unpA_dst_format);
    const std::uint32_t b_x_stride = register_datum_bytes(unpB_dst_format);

    // Channel-1 (source register) strides: one face row and one face.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, cfg::Sec::S0>(0),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(FACE_C_DIM * a_x_stride),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_C_DIM * FACE_R_DIM * a_x_stride),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0>(0),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, cfg::Sec::S0>(0),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(FACE_C_DIM * b_x_stride),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_C_DIM * FACE_R_DIM * b_x_stride),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0>(0));

    // ALU controls owned by the unpack side: dest accumulation width, int8 math, unsigned 8-bit sources,
    // and stochastic rounding (cleared here; _llk_unpack_configure_stoch_rnd_ programs it afterwards).
    const std::uint32_t int8_math = is_int8_or_int32(unpA_dst_format) || is_int8_or_int32(unpB_dst_format);
    sync::mutex::acquire<sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(unpA_src_format == fmt(DataFormat::UInt8)),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(unpB_src_format == fmt(DataFormat::UInt8)),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math));
    sync::mutex::release<sync::Mutex::M0>();

    // Tile descriptors (words 0 and 1). Unpacker 0 takes its X dimension from Tile_x_dim_cntx*.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format_code(unpA_src_format)),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, cfg::Sec::S0>(unpA_num_faces),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1>(format_code(unpB_src_format)),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, cfg::Sec::S1>(unpB_face_r_dim * FACE_C_DIM),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, cfg::Sec::S1>(unpB_num_faces));

    // Unpacker configuration (words 0 and 1 of THCON register 2), identical for both unpackers apart from the
    // output format. Transpose (haloize) is owned by the op init.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0>(format_code(unpA_dst_format)),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, cfg::Sec::S0, 2>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Tileize_mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_Src_Reg_Set_Upd, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_If_Sel, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_rate, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Ovrd_data_format, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_and_interleave, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx0, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx1, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx2, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx3, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S1>(format_code(unpB_dst_format)),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, cfg::Sec::S1, 2>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Tileize_mode, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_Src_Reg_Set_Upd, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_If_Sel, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_rate, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Ovrd_data_format, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_and_interleave, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx0, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx1, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx2, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx3, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, cfg::Sec::S1, 0>());

    // Unpacker-0 SrcA write base per context and per-context face size (datums); unpacker 1 face size.
    const std::uint32_t srca_base = (unpA_face_r_dim == 0) ? 22 * 16 : unpack_detail::SRCA_CONTEXT_ADDRESS;
    const std::uint32_t a_face_dim = unpA_face_r_dim * FACE_C_DIM;
    const std::uint32_t b_face_dim = unpB_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0>(srca_base),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0>(srca_base),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0>(a_face_dim),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0>(a_face_dim),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S1>(b_face_dim),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S1>(b_face_dim),
        cfg::set<cfg::Unpacker[0].ForcedSharedExp, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>());

    // Unpack-thread controls: SrcA set 0 without address override, context offsets back to context 0.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(),
        cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 0>(),
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(),
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());

    unpack_detail::set_face_datum_count(unpA_face_r_dim);
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg                 = hal::cfg;
    namespace sync                = hal::sync;
    constexpr std::uint32_t fpu   = (stoch_rnd_mode == StochRndType::Fpu || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    constexpr std::uint32_t pack  = (stoch_rnd_mode == StochRndType::Pack || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    sync::mutex::acquire<sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack>());
    sync::mutex::release<sync::Mutex::M0>();
}
