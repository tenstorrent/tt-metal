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
#include "llk_defs.h"
#include "llk_memory_checks.h"

using namespace ckernel;

namespace hal_llk::unpack
{

/// Semaphore the unpack TRISC uses to track busy unpacker configuration contexts (ckernel::semaphore::UNPACK_SYNC).
inline constexpr hal::sync::Semaphore ContextSemaphore = hal::sync::Semaphore::S5;

/// SrcA write address (in datums) of unpacker 0: SrcA data starts at row 4 of a bank.
inline constexpr std::uint32_t SrcAWriteBase = 4 * FACE_C_DIM;

/// Number of unpacker configuration contexts the unpack thread double-buffers between.
inline constexpr std::uint32_t NumContexts = 2;

/// Bytes per datum in a source register for the given unpacker output format.
inline constexpr std::uint32_t src_datum_bytes(const std::uint32_t format)
{
    const std::uint32_t low = format & 0x3;
    return (low == static_cast<std::uint32_t>(DataFormat::Float32)) ? 4 : (low == static_cast<std::uint32_t>(DataFormat::Float16)) ? 2 : 1;
}

/// Program one unpacker: tile descriptor, output config, per-context face size and output strides.
template <std::uint32_t U>
inline void configure_unpacker(
    const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    using namespace hal;
    constexpr cfg::Sec S         = (U == 0) ? cfg::Sec::S0 : cfg::Sec::S1;
    const std::uint32_t face_dim = face_r_dim * FACE_C_DIM;

    // Tile descriptor: an uncompressed tile of num_faces planes, each one face of face_dim datums.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, S>(src_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, S>(face_dim),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, S>(num_faces),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.WDim, S, 0>());

    // Output config: register format, default throttling, no transpose/tilize/upsample, no zero compression.
    cfg::write<cfg::Access::TensixCfgUnit>(
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
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, S, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, S, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, S, 0>());

    // Unpacker 0 only: per-context face size (overrides the descriptor X dim) and SrcA write address for both used contexts.
    if constexpr (U == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, S>(face_dim),
            cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, S>(face_dim),
            cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, S, SrcAWriteBase>(),
            cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, S, SrcAWriteBase>());
    }

    // Output (channel 1) strides: one row of FACE_C_DIM datums, one face of FACE_R_DIM rows.
    const std::uint32_t x_stride = src_datum_bytes(dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(x_stride * FACE_C_DIM),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(x_stride * FACE_C_DIM * FACE_R_DIM),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
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
    using namespace hal;
    // Configuration below must not change under an unpacker that is still running.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    hal_llk::unpack::configure_unpacker<0>(unpA_src_format, unpA_dst_format, unpA_face_r_dim, unpA_num_faces);
    hal_llk::unpack::configure_unpacker<1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim, unpB_num_faces);

    // Unpacker 0 adds the channel-1 address counter to the per-context SrcA write address.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>();

    // ALU source formats and accumulation mode (shared state word, written from the unpack thread).
    // UInt8 shares its 4-bit hardware code with Int8; only the full format value tells them apart.
    const std::uint32_t srca_unsigned = (unpA_dst_format == static_cast<std::uint32_t>(DataFormat::UInt8)) ? 1 : 0;
    const std::uint32_t srcb_unsigned = (unpB_dst_format == static_cast<std::uint32_t>(DataFormat::UInt8)) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(unpA_dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(unpB_dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(srca_unsigned),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(srcb_unsigned),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>());

    // Reset all unpacker address counters.
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

    // Start from configuration context 0 for both unpackers.
    unp_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    using namespace hal;
    constexpr std::uint32_t fpu_srnd_en  = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr std::uint32_t pack_srnd_en = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Pack);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd_en>());
}
