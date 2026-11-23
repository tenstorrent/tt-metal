// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole pack LLK functions (PackMode::Default).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/gpr_ops.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/utils/address_modifier.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

namespace hal_llk::pack
{

/// Pack-thread GPR that carries the L1 destination address into the packer config (ckernel::p_gpr_pack::OUTPUT_ADDR).
inline constexpr std::uint32_t OutputAddressGpr = 12;

/// PACR address-modifier slots used by the pack MOP.
inline constexpr std::uint8_t AddrModRowStep = 0; // next group of rows in the same face
inline constexpr std::uint8_t AddrModFaceEnd = 1; // next face
inline constexpr std::uint8_t AddrModTileEnd = 2; // tile finished

/// Packer X stride: datum size of the packer input format (4 for 32-bit, 2 for 16-bit, 1 otherwise).
inline constexpr std::uint32_t pack_src_datum_bytes(const std::uint32_t pack_src_format)
{
    const std::uint32_t low = pack_src_format & 0x3;
    return (low == static_cast<std::uint32_t>(DataFormat::Float32)) ? 4 : (low == static_cast<std::uint32_t>(DataFormat::Float16)) ? 2 : 1;
}

inline constexpr bool is_bfp_format(const std::uint32_t format)
{
    switch (format & DATA_FORMAT_CONFIG_MASK)
    {
        case static_cast<std::uint32_t>(DataFormat::Bfp8):
        case static_cast<std::uint32_t>(DataFormat::Bfp8_b):
        case static_cast<std::uint32_t>(DataFormat::Bfp4):
        case static_cast<std::uint32_t>(DataFormat::Bfp4_b):
        case static_cast<std::uint32_t>(DataFormat::Bfp2):
        case static_cast<std::uint32_t>(DataFormat::Bfp2_b):
            return true;
        default:
            return false;
    }
}

/// Dest rows one PACR moves: one per active Dest-read interface.
inline std::uint32_t rows_per_transfer(const std::uint32_t face_r_dim)
{
    return (face_r_dim < 4) ? face_r_dim : 4;
}

/// Dest-read interface mask for faces shorter than four rows (0 selects all four interfaces).
inline std::uint8_t interface_mask(const std::uint32_t face_r_dim)
{
    return (face_r_dim == 1) ? 0b0001 : (face_r_dim == 2) ? 0b0011 : 0;
}

/// Program the three PACR address modifiers: next rows of a face, next face, end of tile.
template <std::uint32_t RowStep>
inline void configure_address_modifiers()
{
    using hal::cfg::Sec;
    hal::pack::configure_address_modifiers<
        hal::pack::AddressModifier {
            .selection  = Sec::S0,
            .source_row = {.step = RowStep},
        },
        hal::pack::AddressModifier {
            .selection   = Sec::S1,
            .source_row  = {.clear = true},
            .source_face = {.step = 1},
        },
        hal::pack::AddressModifier {
            .selection   = Sec::S2,
            .source_row  = {.clear = true},
            .source_face = {.clear = true},
        }>();
}

} // namespace hal_llk::pack

template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");
    hal_llk::pack::configure_address_modifiers<4>();
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const std::uint32_t num_tiles = 1)
{
    using namespace hal_llk::pack;
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");

    const std::uint32_t rows       = rows_per_transfer(face_r_dim);
    const std::uint8_t interfaces  = interface_mask(face_r_dim);
    constexpr auto datum_override  = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None;

    // Rows within a face, then the next face; the final transfer of the tile closes it.
    hal::pack::DataTransfer row_step {
        .address_modifier = AddrModRowStep,
        .interfaces       = interfaces,
        .datum_override   = datum_override,
    };
    hal::pack::DataTransfer face_end = row_step;
    face_end.address_modifier        = AddrModFaceEnd;
    hal::pack::DataTransfer tile_end = row_step;
    tile_end.address_modifier        = AddrModTileEnd;
    tile_end.boundary                = hal::pack::TileBoundary::Last;

    using Config = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;
    Config config {};
    config.outer_loop.count                                = num_faces * num_tiles;
    config.inner_loop.count                                = face_r_dim / rows;
    config.inner_loop.body_op                              = row_step.get_operation();
    config.inner_loop.last_op_on_nonfinal_outer_iteration = face_end.get_operation();
    config.inner_loop.last_op_on_final_outer_iteration    = tile_end.get_operation();
    hal::mop::program(config);
}

namespace llk_pack_internal_bh
{
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    [[maybe_unused]] const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles)
{
    using namespace hal_llk::pack;
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");
    static_assert(skip_packer_strides, "HAL pack programs the packer strides in _llk_pack_hw_configure_");
    static_assert(!mutex_ADC, "HAL pack does not share the pack ADCs with other threads");

    if constexpr (!skip_addrmod_config)
    {
        // The row step equals the rows one PACR moves (1, 2 or 4).
        switch (rows_per_transfer(face_r_dim))
        {
            case 1:
                configure_address_modifiers<1>();
                break;
            case 2:
                configure_address_modifiers<2>();
                break;
            default:
                configure_address_modifiers<4>();
                break;
        }
    }

    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);

    // Every PACR row moves one row of FACE_C_DIM datums per interface. The X-on-both-channels builder (SETADCXX)
    // places the Channel0 value in the X-end field and the Channel1 value in the X-start field.
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .X<FACE_C_DIM - 1>()
        .channel<hal::AddressChannel::Channel1>()
        .X<0>()
        .apply();
}
} // namespace llk_pack_internal_bh

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t tile_size,
    const std::uint32_t face_r_dim             = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces              = 4,
    [[maybe_unused]] const bool partial_face   = false,
    const std::uint32_t relu_config            = 0)
{
    using namespace hal;
    using namespace hal_llk::pack;
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    // Dest read strides in packer-input-format units: datum, row of FACE_C_DIM datums, face of FACE_R_DIM rows,
    // tile of four faces.
    const std::uint32_t x_stride = pack_src_datum_bytes(pack_src_format);
    const std::uint32_t y_stride = x_stride * FACE_C_DIM;
    const std::uint32_t z_stride = y_stride * FACE_R_DIM;
    const std::uint32_t w_stride = z_stride * TILE_NUM_FACES;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0>(x_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(y_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(w_stride));

    // Packer formats; BFP outputs carry one exponent per 16-datum row, packed into 16-byte words ahead of the data.
    const std::uint32_t exp_rows         = face_r_dim * num_faces;
    const std::uint32_t exp_section_size = is_bfp_format(pack_dst_format) ? ((exp_rows < 16) ? 1 : (exp_rows / 16)) : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_section_size));
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_zero_compress, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Out_data_format, cfg::Sec::S0>(pack_dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg1].In_data_format, cfg::Sec::S0>(pack_src_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg1].Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Source_interface_selection, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Downsample_mask, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Downsample_rate, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0, 0>());

    // Read 32-bit Dest datums when Dest accumulates in 32 bits.
    // UInt8 shares its 4-bit hardware code with Int8; only the full format value tells them apart.
    const std::uint32_t read_unsigned = (pack_src_format == static_cast<std::uint32_t>(DataFormat::UInt8)) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(read_unsigned),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, 0>());

    // One read per XY plane; no edge masking: every row maps to mask set 0, which passes all 16 datums.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xffff>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(), cfg::set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>());

    // ReLU.
    const ckernel::ReluConfig relu = ckernel::ReluConfig::from_packed(relu_config);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(static_cast<std::uint32_t>(relu.get_mode())),
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(relu.get_threshold()));
}

template <
    PackMode pack_mode       = PackMode::Default,
    bool zero_output         = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false,
    bool mutex_ADC           = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    [[maybe_unused]] const bool skip_bh_tilize_workaround)
{
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, true, mutex_ADC>(
        pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

inline void _llk_pack_uninit_()
{
}

template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    using namespace hal;
    using hal_llk::pack::OutputAddressGpr;
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");

    // L1 destination: GPR (two SETDMAREG), then one WRCFG once the packer finished the previous tile.
    gpr_ops::set(hal::gpr<OutputAddressGpr>(), address);
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<OutputAddressGpr>());

    // Select the Dest tile (also separates the WRCFG from the first PACR).
    set_dst_write_addr<mutex_ADC>(tile_index);

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Rewind the face counters for the next tile.
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .apply();
}
