// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole unpack common LLK functions.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

using namespace ckernel;

namespace hal_llk::unpack
{
namespace cfg = hal::cfg;

// Datum byte size of a source-register format: Float32 -> 4, Float16 -> 2, else 1 (keyed on the low 2 bits).
constexpr std::uint32_t src_reg_datum_bytes(const std::uint32_t format)
{
    return ((format & 0x3) == static_cast<std::uint32_t>(DataFormat::Float32))   ? 4
           : ((format & 0x3) == static_cast<std::uint32_t>(DataFormat::Float16)) ? 2
                                                                                 : 1;
}

constexpr bool is_int8_math_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0xF;
    return f == static_cast<std::uint32_t>(DataFormat::Int8) || f == static_cast<std::uint32_t>(DataFormat::Int32);
}

constexpr bool is_uint8_format(const std::uint32_t format)
{
    return (format & 0xF) == static_cast<std::uint32_t>(DataFormat::UInt8);
}

// Unpacker 0 writes SrcA starting at row 4 of the selected set; the set itself comes from the write address.
// Observed on hardware: with row 0 and no set override the math thread never receives the SrcA bank.
constexpr std::uint32_t SRCA_WRITE_BASE_ROW = 4;

// Wait until every unpacker configuration context handed to the unpackers has been released.
inline void wait_for_idle()
{
    while (hal::sync::semaphore::read<hal::sync::Semaphore::S5>() > 0)
    {
    }
}

// Select configuration context 0 for both unpackers.
inline void reset_config_context()
{
    unp_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

// Flip both unpackers to the other configuration context.
inline void switch_config_context()
{
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
}

// Last datum index of one unpack: face_r_dim rows of FACE_C_DIM datums.
constexpr std::uint32_t unpacker_x_end(const std::uint32_t face_r_dim)
{
    return (face_r_dim == 0) ? 1 : (face_r_dim * FACE_C_DIM) - 1;
}

// Program the datum range [0, x_end] of one unpack. Observed on hardware: the end value must be given as the
// Channel0 X of the SETADCXX builder and the start as Channel1 X (the reverse ordering hangs the unpacker).
template <hal::AddressCounterClient... Clients>
inline void set_unpacker_x_range(const std::uint32_t face_r_dim)
{
    hal::runtime_address_counters.template client<Clients...>()
        .template channel<hal::AddressChannel::Channel0>()
        .X(unpacker_x_end(face_r_dim))
        .template channel<hal::AddressChannel::Channel1>()
        .X(0)
        .apply();
}

// Program the per-unpack datum count of both unpackers.
inline void config_unpacker_x_end(const std::uint32_t face_r_dim)
{
    set_unpacker_x_range<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>(face_r_dim);
}

// Reset the Y/Z/W address counters of both unpackers on both channels.
inline void reset_unpacker_yzw_counters()
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
    using namespace hal_llk::unpack;

    // All contexts must be released before the configuration is replaced.
    wait_for_idle();

    // Reset every unpacker address counter.
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

    // Channel-1 (register side) strides: one face row and one face, in bytes of the register format.
    const std::uint32_t unpA_x_stride = src_reg_datum_bytes(unpA_dst_format);
    const std::uint32_t unpB_x_stride = src_reg_datum_bytes(unpB_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(FACE_C_DIM * unpA_x_stride),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_R_DIM * FACE_C_DIM * unpA_x_stride),
        cfg::set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(FACE_C_DIM * unpB_x_stride),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_R_DIM * FACE_C_DIM * unpB_x_stride),
        cfg::set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());

    // ALU source/accumulation formats, programmed from the unpacker side (shared word, Tensix RMW).
    const bool int8_math = is_int8_math_format(unpA_dst_format) || is_int8_math_format(unpB_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(is_uint8_format(unpA_dst_format) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(is_uint8_format(unpB_dst_format) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(unpA_dst_format & 0xF),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(unpB_dst_format & 0xF),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));

    // FP8 (E4M3) input uses the 4-bit exponent mode of the unpacker.
    const bool lf8_4b_exp = (unpA_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0>(lf8_4b_exp ? 1 : 0);

    // Tile descriptors. SrcA X dim comes from the per-context Tile_x_dim override, so its descriptor X dim is 0.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(unpA_src_format & 0xF),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, cfg::Sec::S0>(unpA_num_faces),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1>(unpB_src_format & 0xF),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, cfg::Sec::S1>(unpB_face_r_dim * FACE_C_DIM),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, cfg::Sec::S1>(unpB_num_faces));

    // Unpacker output configuration (first two words): output format, throttling, no transpose, uncompressed contexts.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0>(unpA_dst_format & 0xF),
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
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S1>(unpB_dst_format & 0xF),
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
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx3, cfg::Sec::S1, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx2, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx3, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, cfg::Sec::S1, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx2, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx3, cfg::Sec::S1, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, cfg::Sec::S1, 0>());

    // Datum count per unpack: one face row band (face_r_dim rows of FACE_C_DIM datums).
    set_unpacker_x_range<hal::AddressCounterClient::Unpacker0>(unpA_face_r_dim);
    set_unpacker_x_range<hal::AddressCounterClient::Unpacker1>(unpB_face_r_dim);

    // SrcA write address (both contexts) and per-context SrcA face size in L1.
    const std::uint32_t srca_dest_address = SRCA_WRITE_BASE_ROW * FACE_C_DIM;
    const std::uint32_t face_dim          = unpA_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0>(srca_dest_address),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0>(srca_dest_address),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0>(face_dim),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0>(face_dim));

    // SrcA set selection follows the upper bits of the unpacker write address.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());

    // Final SrcA write address = per-context Dest address + channel-1 address counter.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>();

    reset_config_context();
}

template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg                  = hal::cfg;
    constexpr std::uint32_t fpu    = (stoch_rnd_mode == StochRndType::Fpu || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    constexpr std::uint32_t packer = (stoch_rnd_mode == StochRndType::Pack || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, packer>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, packer>());
}
