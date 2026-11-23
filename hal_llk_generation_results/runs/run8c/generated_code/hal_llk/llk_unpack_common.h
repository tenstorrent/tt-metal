// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the common unpacker configuration (Blackhole).

#pragma once

#include <cstdint>

#include "hal_llk_common.h"
#include "llk_memory_checks.h"

namespace hal_llk
{

// Tile descriptor of one unpacker (Sec::S0 = unpacker 0, Sec::S1 = unpacker 1).
template <hal::cfg::Sec S>
inline void program_unpack_tile_descriptor(const std::uint32_t src_format, const std::uint32_t x_dim, const std::uint32_t num_faces)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, S>(format_code(src_format)),
        set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, S, 1>(),
        set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, S, 0>(),
        set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, S>(x_dim),
        set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, S, 1>(),
        set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, S>(num_faces));
}

// Unpacker configuration of one unpacker: output format, high-performance throttle mode,
// uncompressed input in every context, no per-context format override, no transpose.
template <hal::cfg::Sec S>
inline void program_unpack_config(const std::uint32_t dst_format)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::Thcon[cfg::Reg2].Out_data_format, S>(format_code(dst_format)),
        set<cfg::Thcon[cfg::Reg2].Throttle_mode, S, 2>(),
        set<cfg::Thcon[cfg::Reg2].Context_count, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Haloize_mode, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Tileize_mode, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_Src_Reg_Set_Upd, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_If_Sel, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Upsample_rate, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Ovrd_data_format, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Upsample_and_interleave, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx0, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx1, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx2, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx3, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, S, 1>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx2, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx3, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx4, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx5, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx6, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx7, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Force_shared_exp, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Context_count_non_log2, S, 0>(),
        set<cfg::Thcon[cfg::Reg2].Context_count_non_log2_en, S, 0>());
}

// SrcA write offset of the data contexts (4 rows) and L1 X dimension (face size in datums) of unpacker 0.
inline constexpr std::uint32_t UNPACK_SRCA_CONTEXT_OFFSET = 0x40;

template <hal::cfg::Sec S>
inline void program_unpack_context_dims(const std::uint32_t face_dim)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, S, UNPACK_SRCA_CONTEXT_OFFSET>(),
        set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, S, UNPACK_SRCA_CONTEXT_OFFSET>(),
        set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, S>(face_dim),
        set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, S>(face_dim));
}

// Datum count of one face for the selected unpackers (SETADCXX: X end and X start = 0).
// Observed on hardware: the HAL builder places the Channel0 X value in the X-end field of SETADCXX
// and the Channel1 X value in the X-start field (see the final report).
template <hal::AddressCounterClient... Clients>
inline void program_unpack_x_end(const std::uint32_t face_r_dim)
{
    const std::uint32_t x_end = (face_r_dim * FACE_C_DIM) - 1;
    hal::runtime_address_counters.client<Clients...>()
        .template channel<hal::AddressChannel::Channel0>()
        .X(x_end)
        .template channel<hal::AddressChannel::Channel1>()
        .X(0)
        .apply();
}

// Select unpacker configuration context `context` for both unpackers (added to the UNPACR context ID).
inline void select_unpack_config_context(const std::uint32_t context)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    if (context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
}

} // namespace hal_llk

/**
 * @brief Configure the unpacker hardware for both operands A and B.
 *
 * Programs the per-operand tile descriptors, unpacker output formats, channel-1 strides, the SrcA/SrcB ALU
 * formats and FP32/INT8 math modes, the per-context X dimension of unpacker 0, and resets the address
 * counters and the configuration context.
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
    namespace cfg = hal::cfg;
    using cfg::Access;
    using cfg::Sec;
    using cfg::set;
    using namespace hal::sync;

    // All configuration contexts must be released before the unpacker is reprogrammed.
    while (semaphore::read<hal_llk::UNPACK_SYNC_SEMAPHORE>() > 0)
    {
    }
    wait::stall<StallTarget::Config, StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle>();

    // Reset unpacker Y/Z/W address counters (X start/end are programmed below with SETADCXX).
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

    // Channel-1 (register side) strides of unpacker 0: one face row and one face of the register datum size.
    // Unpacker 1 only needs the face stride.
    const std::uint32_t unpA_x_stride = hal_llk::register_datum_bytes(unpA_dst_format);
    const std::uint32_t unpB_x_stride = hal_llk::register_datum_bytes(unpB_dst_format);
    cfg::write<Access::TensixCfgUnit>(
        set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, Sec::S0, 0>(),
        set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, Sec::S0>(FACE_C_DIM * unpA_x_stride),
        set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, Sec::S0>(FACE_R_DIM * FACE_C_DIM * unpA_x_stride),
        set<cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, Sec::S0, 0>(),
        set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, Sec::S0, 0>(),
        set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, Sec::S0, 0>(),
        set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, Sec::S0>(FACE_R_DIM * FACE_C_DIM * unpB_x_stride),
        set<cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, Sec::S0, 0>());

    // Dest modes and the Dest ALU format. The SrcA/SrcB ALU formats are implied by the unpackers on Blackhole.
    const bool int8_math              = hal_llk::is_int8_math_format(unpA_dst_format) || hal_llk::is_int8_math_format(unpB_dst_format);
    const std::uint32_t dest_format   = (is_fp32_dest_acc_en && !int8_math) ? static_cast<std::uint32_t>(DataFormat::Float32) : hal_llk::format_code(unpA_dst_format);
    const std::uint32_t srca_unsigned = ((unpA_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::UInt8)) ? 1 : 0;
    const std::uint32_t srcb_unsigned = ((unpB_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::UInt8)) ? 1 : 0;
    cfg::write<Access::TensixCfgUnit>(
        set<cfg::AluFormatSpecReg0::SrcAUnsigned, Sec::S0>(srca_unsigned),
        set<cfg::AluFormatSpecReg0::SrcBUnsigned, Sec::S0>(srcb_unsigned),
        set<cfg::AluFormatSpecReg2::Dstacc, Sec::S0>(dest_format),
        set<cfg::AluAccCtrl::Fp32_enabled, Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        set<cfg::AluAccCtrl::SFPU_Fp32_enabled, Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        set<cfg::AluAccCtrl::INT8_math_enabled, Sec::S0>(int8_math ? 1 : 0));

    // Tile descriptors. Unpacker 0 takes its X dimension from the per-context Tile_x_dim (explicit-context UNPACR).
    hal_llk::program_unpack_tile_descriptor<Sec::S0>(unpA_src_format, 0, unpA_num_faces);
    hal_llk::program_unpack_tile_descriptor<Sec::S1>(unpB_src_format, unpB_face_r_dim * FACE_C_DIM, unpB_num_faces);

    hal_llk::program_unpack_config<Sec::S0>(unpA_dst_format);
    hal_llk::program_unpack_config<Sec::S1>(unpB_dst_format);

    // Unpacker 0: per-context face size, SrcA write offset, write address = context offset + address counter.
    hal_llk::program_unpack_context_dims<Sec::S0>(unpA_face_r_dim * FACE_C_DIM);
    cfg::write<Access::TensixCfgUnit>(set<cfg::Unpacker[0].AddDestAddrCntr, Sec::S0, 1>());
    cfg::write<Access::TensixCfgUnit>(set<cfg::Unpacker[1].AddDestAddrCntr, Sec::S0, 0>());

    hal_llk::program_unpack_x_end<hal::AddressCounterClient::Unpacker0>(unpA_face_r_dim);
    hal_llk::program_unpack_x_end<hal::AddressCounterClient::Unpacker1>(unpB_face_r_dim);

    // SrcA write set follows the unpacker write address.
    cfg::write<Access::TensixCfgUnit>(set<cfg::SrcASet::Base, Sec::S0, 0>(), set<cfg::SrcASet::SetOvrdWithAddr, Sec::S0, 1>());

    // Start from configuration context 0.
    unp_cfg_context = 0;
    hal_llk::select_unpack_config_context(0);
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg = hal::cfg;
    using cfg::set;
    constexpr std::uint32_t fpu_srnd  = (stoch_rnd_mode == StochRndType::Fpu || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    constexpr std::uint32_t pack_srnd = (stoch_rnd_mode == StochRndType::Pack || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd>(),
        set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd>(),
        set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd>());
}
