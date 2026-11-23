// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the unpack-side common LLK functions.

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk::unpack
{

// Configuration context currently free for the RISC to program (the unpacker uses the other one).
// Context c lives in Unpacker[u].Cntx[c]; UnpackMiscCfg::CfgContextOffset_* selects which one UNPACR uses.
inline std::uint32_t next_context = 0;

// Wait until the unpacker has released every configuration context it was given.
inline void wait_for_idle()
{
    while (hal::sync::semaphore::read<SEM_UNPACK_SYNC>() > 0)
    {
    }
}

// Point both unpackers at configuration context 0 and hand context 0 to the RISC.
inline void reset_config_context()
{
    next_context = 0;
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::UnpackMiscCfg::CfgContextOffset_0, hal::cfg::Sec::S0, 0>(),
        hal::cfg::set<hal::cfg::UnpackMiscCfg::CfgContextOffset_1, hal::cfg::Sec::S0, 0>());
}

// Program one unpacker: tile descriptor (L1 side), output format/mode and srcA/srcB-side strides.
template <hal::cfg::Sec S>
inline void configure_unpacker(
    const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces, const bool transpose_xy)
{
    const std::uint32_t face_dim     = face_r_dim * FACE_COLS;
    const std::uint32_t ch1_x_stride = reg_datum_bytes(dst_format);

    // L1 tile descriptor: one face is one Z plane of face_dim datums (Y = 1).
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.InDataFormat, S>(hw_format(src_format)),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.Uncompressed, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.BlobsPerXyPlane, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.XDim, S>(face_dim),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.YDim, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.ZDim, S>(num_faces));

    // Unpacker output format and mode: high-perf throttle, no tilize/upsample, uncompressed in every context.
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Out_data_format, S>(hw_format(dst_format)),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Throttle_mode, S, 2>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Context_count, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Haloize_mode, S>(transpose_xy ? 1u : 0u),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Tileize_mode, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Unpack_Src_Reg_Set_Upd, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Unpack_If_Sel, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Upsample_rate, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Ovrd_data_format, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Upsample_and_interleave, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Shift_amount_cntx0, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Shift_amount_cntx1, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx0, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx1, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx2, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx3, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Unpack_if_sel_cntx0, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Unpack_if_sel_cntx1, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Force_shared_exp, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Context_count_non_log2_en, S, 0>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx4, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx5, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx6, S, 1>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg2].Disable_zero_compress_cntx7, S, 1>());

    // Per-context face width (used because UNPACR runs in multi-context mode).
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg5].Tile_x_dim_cntx0, S>(face_dim), hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg5].Tile_x_dim_cntx1, S>(face_dim));

    // Source register (channel 1) strides: one face row per Y step, one face per Z step.
    constexpr std::uint32_t U = (S == hal::cfg::Sec::S0) ? 0 : 1;
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::Unpacker[U].AddrCtrl[hal::cfg::UnpackerReg::Reg1].Ystride, hal::cfg::Sec::S0>(FACE_COLS * ch1_x_stride),
        hal::cfg::set<hal::cfg::Unpacker[U].AddrCtrl[hal::cfg::UnpackerReg::Reg1].Zstride, hal::cfg::Sec::S0>(FACE_ROWS * FACE_COLS * ch1_x_stride),
        hal::cfg::set<hal::cfg::Unpacker[U].AddrCtrl[hal::cfg::UnpackerReg::Reg1].Wstride, hal::cfg::Sec::S0, 0>());
}

// Datum range of one UNPACR on both unpackers: channel-0 X is the first datum, channel-1 X the last.
template <std::uint32_t XEnd>
inline void set_unpacker_x_range()
{
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X<0>()
        .apply();
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel1>()
        .X<XEnd>()
        .apply();
}

// One face (face_r_dim rows of 16 datums) per UNPACR.
inline void config_unpacker_x_end(const std::uint32_t face_r_dim)
{
    // The runtime-value address-counter builder hangs the kernel when it targets both channels
    // (see the report), so each supported face height gets a compile-time encoding.
    switch (face_r_dim)
    {
        case 1:
            set_unpacker_x_range<1 * FACE_COLS - 1>();
            break;
        case 2:
            set_unpacker_x_range<2 * FACE_COLS - 1>();
            break;
        case 4:
            set_unpacker_x_range<4 * FACE_COLS - 1>();
            break;
        case 8:
            set_unpacker_x_range<8 * FACE_COLS - 1>();
            break;
        default:
            set_unpacker_x_range<FACE_ROWS * FACE_COLS - 1>();
            break;
    }
}

} // namespace hal_llk::unpack

/**
 * @brief Configure the unpacker hardware for both operands A and B (formats, face geometry, contexts).
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
    using namespace hal_llk;
    namespace cfg = hal::cfg;

    // All configuration contexts must be released before the unpacker is reprogrammed.
    unpack::wait_for_idle();
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::Unpacker0Idle | hal::sync::StallCondition::Unpacker1Idle>();
    hal::sync::semaphore::init<0 /* Initial */, 2 /* Maximum */, SEM_UNPACK_SYNC>();

    // Reset the unpacker address counters (the X range is programmed below).
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

    unpack::configure_unpacker<cfg::Sec::S0>(unpA_src_format, unpA_dst_format, unpA_face_r_dim, unpA_num_faces, false);
    unpack::configure_unpacker<cfg::Sec::S1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim, unpB_num_faces, false);

    // ALU source formats, dest accumulation mode and integer math mode (owned by the unpack side on Blackhole).
    const bool int8_math = is_int8_or_int32(unpA_dst_format) || is_int8_or_int32(unpB_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(hw_format(unpA_dst_format)),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(hw_format(unpB_dst_format)),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1u : 0u));

    // SrcA is written past the 4 halo rows (row 4 onwards) in both contexts; unpacker 0 adds the
    // channel-1 address counter to this per-context destination address.
    constexpr std::uint32_t SRCA_FIRST_ROW_DATUM = 4 * FACE_COLS;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, SRCA_FIRST_ROW_DATUM>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, SRCA_FIRST_ROW_DATUM>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>();
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());

    unpack::config_unpacker_x_end(unpA_face_r_dim);

    unpack::reset_config_context();
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg = hal::cfg;
    constexpr std::uint32_t fpu_srnd_en  = (stoch_rnd_mode == StochRndType::All || stoch_rnd_mode == StochRndType::Fpu) ? 1 : 0;
    constexpr std::uint32_t pack_srnd_en = (stoch_rnd_mode == StochRndType::All || stoch_rnd_mode == StochRndType::Pack) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd_en>());
}
