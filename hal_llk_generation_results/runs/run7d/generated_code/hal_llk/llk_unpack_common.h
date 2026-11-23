// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the unpacker hardware configuration shared by the unpack LLKs.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal_llk_common.h"
#include "llk_memory_checks.h"
#include "llk_defs.h"

using namespace ckernel;

namespace hal_llk::unpack_detail
{

namespace cfg = hal::cfg;

// SrcA is written by unpacker 0 starting four rows (64 datums) into the bank: the first rows are the
// math halo, so the FPU addresses SrcA relative to this offset.
constexpr std::uint32_t SRCA_WRITE_OFFSET = 4 * FACE_C_DIM;

// Drain both unpackers so their configuration can be rewritten safely.
inline void wait_for_unpackers_idle()
{
    using namespace hal::sync;
    wait::stall<StallTarget::Config, StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle>();
}

// Output (channel 1) strides of one unpacker: one face row and one face, in register-format units.
template <std::uint32_t Unpacker>
inline void configure_output_strides(const std::uint32_t dst_format)
{
    const std::uint32_t x_stride = register_datum_bytes(dst_format);
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride = FACE_R_DIM * y_stride;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[Unpacker].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Unpacker[Unpacker].AddrCtrl[cfg::UnpackerReg::Reg1].Ystride, cfg::Sec::S0>(y_stride),
        cfg::set<cfg::Unpacker[Unpacker].AddrCtrl[cfg::UnpackerReg::Reg1].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Unpacker[Unpacker].AddrCtrl[cfg::UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
}

// Tile descriptor and unpacker configuration words of one unpacker (THCON section S0 = unpacker 0, S1 = unpacker 1).
template <cfg::Sec Section>
inline void configure_unpacker(const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    const std::uint32_t face_datums = face_r_dim * FACE_C_DIM;

    // Tile descriptor: uncompressed input, one face per Z step.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, Section>(format_code(src_format)),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, Section>(face_datums),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.YDim, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.ZDim, Section>(num_faces));

    // Unpacker configuration: output format, default throttling, no transforms, no zero compression.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, Section>(format_code(dst_format)),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, Section, 2>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Haloize_mode, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Tileize_mode, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_Src_Reg_Set_Upd, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_If_Sel, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_rate, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Ovrd_data_format, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Upsample_and_interleave, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx0, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx1, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx2, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Shift_amount_cntx3, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx0, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx1, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx2, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx3, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx0, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx1, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx2, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx3, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Force_shared_exp, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count_non_log2, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Context_count_non_log2_en, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx4, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx5, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx6, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Disable_zero_compress_cntx7, Section, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx4, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx5, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx6, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Unpack_if_sel_cntx7, Section, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg2].Metadata_x_end, Section, 0>());

    // Per-context face size in L1 (datums); one Z step of the input counter advances by one face.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, Section>(face_datums), cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, Section>(face_datums));
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
    namespace cfg = hal::cfg;
    using namespace hal_llk;
    using namespace hal_llk::unpack_detail;

    wait_for_unpackers_idle();

    // Start from zeroed X/Y/Z/W address counters on both channels of both unpackers.
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

    configure_output_strides<0>(unpA_dst_format);
    configure_output_strides<1>(unpB_dst_format);

    // ALU source formats and Dest accumulation mode as seen by the FPU.
    const bool int8_math = is_int8_math_format(unpA_dst_format) || is_int8_math_format(unpB_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(is_format(unpA_dst_format, DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(is_format(unpB_dst_format, DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0>(format_code(unpA_dst_format)),
        cfg::set<cfg::AluFormatSpecReg1::SrcB, cfg::Sec::S0>(format_code(unpB_dst_format)),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));

    configure_unpacker<cfg::Sec::S0>(unpA_src_format, unpA_dst_format, unpA_face_r_dim, unpA_num_faces);
    configure_unpacker<cfg::Sec::S1>(unpB_src_format, unpB_dst_format, unpB_face_r_dim, unpB_num_faces);

    // SrcA write position: per-context destination address combined with the channel-1 counters.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, SRCA_WRITE_OFFSET>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, SRCA_WRITE_OFFSET>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].ForcedSharedExp, cfg::Sec::S0, 0>(), cfg::set<cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>());
    // SrcA set index taken from the upper write-address bits (CFG catalog: SrcASet::SetOvrdWithAddr).
    cfg::write<cfg::Access::TensixCfgUnit, cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>();

    // Configuration contexts: always use context 0 with no offset.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg = hal::cfg;
    constexpr std::uint32_t fpu_srnd  = (stoch_rnd_mode == StochRndType::Fpu || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    constexpr std::uint32_t pack_srnd = (stoch_rnd_mode == StochRndType::Pack || stoch_rnd_mode == StochRndType::All) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd>());
}
