// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk
{

// Reset the unpacker configuration-context offsets so UNPACR context 0 resolves to context 0.
inline void unpack_reset_config_context()
{
    namespace cfg = hal::cfg;
    unp_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

// Program the static per-unpacker state: tile descriptor, unpack config and per-context face size.
// Sec::S0 is unpacker 0 (SrcA), Sec::S1 is unpacker 1 (SrcB).
template <hal::cfg::Sec S>
inline void unpack_configure_unpacker(const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    namespace cfg          = hal::cfg;
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    const std::uint32_t face_dim = face_r_dim * FACE_C_DIM;

    // Tile descriptor: uncompressed face_r_dim x 16 faces, num_faces of them along Z.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<TileDsc.InDataFormat, S>(hw_format(src_format)),
        cfg::set<TileDsc.Uncompressed, S, 1>(),
        cfg::set<TileDsc.BlobsPerXyPlane, S, 0>(),
        cfg::set<TileDsc.XDim, S>(face_dim),
        cfg::set<TileDsc.YDim, S, 1>(),
        cfg::set<TileDsc.ZDim, S>(num_faces),
        cfg::set<TileDsc.WDim, S, 0>(),
        cfg::set<TileDsc.BlobsYStartLo, S, 0>(),
        cfg::set<TileDsc.BlobsYStartHi, S, 0>(),
        cfg::set<TileDsc.DigestType, S, 0>(),
        cfg::set<TileDsc.DigestSize, S, 0>());

    // Unpacker control: output format, high-throughput L1 reads, no tilize/upsample/shift,
    // every context uncompressed, no wraparound.
    constexpr auto R2 = cfg::Thcon[cfg::Reg2];
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<R2.Out_data_format, S>(hw_format(dst_format)),
        cfg::set<R2.Throttle_mode, S, 2>(),
        cfg::set<R2.Context_count, S, 0>(),
        cfg::set<R2.Haloize_mode, S, 0>(),
        cfg::set<R2.Tileize_mode, S, 0>(),
        cfg::set<R2.Unpack_Src_Reg_Set_Upd, S, 0>(),
        cfg::set<R2.Unpack_If_Sel, S, 0>(),
        cfg::set<R2.Upsample_rate, S, 0>(),
        cfg::set<R2.Ovrd_data_format, S, 0>(),
        cfg::set<R2.Upsample_and_interleave, S, 0>(),
        cfg::set<R2.Shift_amount_cntx0, S, 0>(),
        cfg::set<R2.Shift_amount_cntx1, S, 0>(),
        cfg::set<R2.Shift_amount_cntx2, S, 0>(),
        cfg::set<R2.Shift_amount_cntx3, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<R2.Disable_zero_compress_cntx0, S, 1>(),
        cfg::set<R2.Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<R2.Disable_zero_compress_cntx2, S, 1>(),
        cfg::set<R2.Disable_zero_compress_cntx3, S, 1>(),
        cfg::set<R2.Unpack_if_sel_cntx0, S, 0>(),
        cfg::set<R2.Unpack_if_sel_cntx1, S, 0>(),
        cfg::set<R2.Unpack_if_sel_cntx2, S, 0>(),
        cfg::set<R2.Unpack_if_sel_cntx3, S, 0>(),
        cfg::set<R2.Force_shared_exp, S, 0>(),
        cfg::set<R2.Context_count_non_log2, S, 0>(),
        cfg::set<R2.Context_count_non_log2_en, S, 0>(),
        cfg::set<R2.Disable_zero_compress_cntx4, S, 1>(),
        cfg::set<R2.Disable_zero_compress_cntx5, S, 1>(),
        cfg::set<R2.Disable_zero_compress_cntx6, S, 1>(),
        cfg::set<R2.Disable_zero_compress_cntx7, S, 1>(),
        cfg::set<R2.Unpack_if_sel_cntx4, S, 0>(),
        cfg::set<R2.Unpack_if_sel_cntx5, S, 0>(),
        cfg::set<R2.Unpack_if_sel_cntx6, S, 0>(),
        cfg::set<R2.Unpack_if_sel_cntx7, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R2.Unpack_limit_address, S, 0>(), cfg::set<R2.Unpack_fifo_size, S, 0>());

    // Per-context source face size (used because UNPACR selects explicit contexts) and, for
    // unpacker 0, the SrcA start address of each context.
    constexpr auto R5 = cfg::Thcon[cfg::Reg5];
    if constexpr (S == cfg::Sec::S0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<R5.Dest_cntx0_address, S, SRCA_UNPACK_START>(), cfg::set<R5.Dest_cntx1_address, S, SRCA_UNPACK_START>());
    }
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R5.Tile_x_dim_cntx0, S>(face_dim), cfg::set<R5.Tile_x_dim_cntx1, S>(face_dim));
}

// Output-side (channel 1) strides for unpacker U, derived from the register format it writes.
template <std::uint32_t U>
inline void unpack_configure_output_strides(const std::uint32_t dst_format)
{
    namespace cfg                 = hal::cfg;
    const std::uint32_t row_bytes = FACE_C_DIM * datum_bytes(dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(row_bytes),
        cfg::set<cfg::Unpacker[U].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(FACE_R_DIM * row_bytes));
}

} // namespace hal_llk

/**
 * @brief Configure both unpackers (formats, tile geometry, ALU source formats and dest accumulation mode).
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

    // Reprogram only once both unpackers have drained.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // Reset the unpacker address counters (X start, Y, Z, W on both channels).
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X<0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .apply();

    hal_llk::unpack_configure_output_strides<0>(unpA_dst_format);
    hal_llk::unpack_configure_output_strides<1>(unpB_dst_format);

    // ALU source formats (programmed unpack-side on Blackhole), integer signedness and dest
    // accumulation mode. Stochastic rounding defaults to off; see _llk_unpack_configure_stoch_rnd_.
    const std::uint32_t int8_math = (hal_llk::is_int8_or_int32(unpA_dst_format) || hal_llk::is_int8_or_int32(unpB_dst_format)) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(unpA_src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(unpB_src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(hal_llk::hw_format(unpA_dst_format)),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(hal_llk::hw_format(unpB_dst_format)),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math));

    hal_llk::unpack_configure_unpacker<cfg::Sec::S0>(unpA_src_format, unpA_dst_format, unpA_face_r_dim, unpA_num_faces);
    hal_llk::unpack_configure_unpacker<cfg::Sec::S1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim, unpB_num_faces);

    // Unpacker 0 output address = per-context dest address + channel-1 address counter.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>();

    hal_llk::unpack_reset_config_context();
}

/**
 * @brief Enable stochastic rounding on the units selected by stoch_rnd_mode.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg           = hal::cfg;
    constexpr std::uint32_t fpu  = (stoch_rnd_mode == StochRndType::Fpu || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    constexpr std::uint32_t pack = (stoch_rnd_mode == StochRndType::Pack || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack>());
}
