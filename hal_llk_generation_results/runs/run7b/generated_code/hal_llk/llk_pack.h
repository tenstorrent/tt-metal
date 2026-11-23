// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the packer (spec: llk_lib/llk_pack.h), PackMode::Default.
//
// Model: one PACR moves one 16-datum Dest row through a single Dest-read interface. A face is
// face_r_dim PACRs (Y steps one row), faces are 16 Dest rows apart (Z), tiles are a 32x32 tile apart (W).

#pragma once

#include <cstdint>

#include "llk_pack_common.h"

namespace hal_llk::pack
{

// ADDR_MOD_PACK slots: next row of the face / first row of the next face.
constexpr std::uint8_t PACK_ADDR_MOD_ROW  = 0;
constexpr std::uint8_t PACK_ADDR_MOD_FACE = 1;

constexpr hal::pack::DataTransfer row_transfer(const std::uint8_t addr_mod, const hal::pack::TileBoundary boundary)
{
    return hal::pack::DataTransfer {
        .address_modifier = addr_mod,
        .interfaces       = 0b0001,
        .boundary         = boundary,
    };
}

constexpr std::uint32_t PACK_ROW       = row_transfer(PACK_ADDR_MOD_ROW, hal::pack::TileBoundary::NotLast).get_operation();
constexpr std::uint32_t PACK_FACE_END  = row_transfer(PACK_ADDR_MOD_FACE, hal::pack::TileBoundary::NotLast).get_operation();
constexpr std::uint32_t PACK_TILE_LAST = row_transfer(PACK_ADDR_MOD_FACE, hal::pack::TileBoundary::Last).get_operation();

template <cfg::Sec S>
inline void write_pack_counters_section(const std::uint32_t reads_per_xy_plane)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, S, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, S>(reads_per_xy_plane),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, S, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, S, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, S, 0>());
}

template <std::uint32_t... Sections>
inline void write_pack_counters(const std::uint32_t reads_per_xy_plane, std::integer_sequence<std::uint32_t, Sections...>)
{
    (write_pack_counters_section<static_cast<cfg::Sec>(Sections)>(reads_per_xy_plane), ...);
}

} // namespace hal_llk::pack

template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    constexpr hal::pack::AddressModifier next_row {
        .selection       = hal::cfg::Sec::S0,
        .source_row      = {.step = 1},
        .destination_row = {.step = 1},
    };
    constexpr hal::pack::AddressModifier next_face {
        .selection       = hal::cfg::Sec::S1,
        .source_row      = {.clear = true},
        .destination_row = {.clear = true},
        .source_face     = {.step = 1},
    };
    hal::pack::configure_address_modifiers<next_row, next_face>();
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const std::uint32_t num_tiles  = 1)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    static_assert(!zero_output, "zero_output is not implemented");
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    MopConfig<MopTemplate::Template1> config {};
    config.outer_loop.count                               = num_faces * num_tiles;
    config.inner_loop.count                               = face_r_dim;
    config.inner_loop.body_op                             = hal_llk::pack::PACK_ROW;
    config.inner_loop.last_op_on_nonfinal_outer_iteration = hal_llk::pack::PACK_FACE_END;
    config.inner_loop.last_op_on_final_outer_iteration    = hal_llk::pack::PACK_TILE_LAST;
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
    static_assert(skip_packer_strides, "Packer strides are owned by _llk_pack_hw_configure_");
    static_assert(!mutex_ADC, "mutex_ADC is not implemented");
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);

    // One PACR reads one 16-datum row (Channel0 value = last datum, Channel1 value = first; see unpack).
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
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const std::uint32_t relu_config                 = 0)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    namespace cfg = hal::cfg;
    using namespace hal_llk;

    pack::wait_packer_idle_for_config();

    // Dest-side (channel 0) strides: one row, one face, one 32x32 tile.
    const std::uint32_t x_stride = datum_bytes(pack_src_format);
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride = FACE_R_DIM * y_stride;
    const std::uint32_t w_stride = TILE_NUM_FACES * z_stride;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(y_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(w_stride));

    // Packer formats and output layout. BFP exponent section covers the whole tile (16B granules).
    const bool exp_section_unused = format_code(pack_dst_format) == static_cast<std::uint32_t>(DataFormat::Lf8) ||
                                    format_code(pack_dst_format) == format_code(static_cast<std::uint32_t>(DataFormat::Int8));
    const std::uint32_t exp_section_size = exp_section_unused ? 0 : (partial_face ? 1 : num_faces);
    const bool exp_threshold             = is_fp32_dest_acc_en && is_bfp_a_format(pack_dst_format);
    const bool lf8_4b_exp                = (pack_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_section_size),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_zero_compress, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Out_data_format, cfg::Sec::S0>(format_code(pack_dst_format)),
        cfg::set<cfg::Thcon[cfg::Reg1].In_data_format, cfg::Sec::S0>(format_code(pack_src_format)),
        cfg::set<cfg::Thcon[cfg::Reg1].Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Source_interface_selection, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0>(exp_threshold ? 1 : 0),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold, cfg::Sec::S0>(exp_threshold ? 113 : 0),
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0>(lf8_4b_exp ? 1 : 0));

    // How Dest is read: 32-bit containers when Dest accumulates in FP32 or holds 32-bit data.
    const bool read_32b = is_fp32_dest_acc_en || is_32bit_format(pack_src_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(read_32b ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(pack_src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>(!read_32b && format_code(pack_src_format) == format_code(static_cast<std::uint32_t>(DataFormat::Int8)) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, 0>());

    // ReLU from the packed relu config (mode in the low bits, threshold in the high half).
    const ckernel::ReluConfig relu = ckernel::ReluConfig::from_packed(relu_config);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(static_cast<std::uint32_t>(relu.get_mode())),
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(relu.get_threshold()));

    // Edge masks pass every datum; all row sets map to mask 0; one Dest read per XY plane.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeOffsetSec1::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeOffsetSec2::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeOffsetSec3::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 4 /* word count */>(std::array<std::uint32_t, 4> {0, 0, 0, 0});
    pack::write_pack_counters(1 /*reads_per_xy_plane*/, std::integer_sequence<std::uint32_t, 0, 1, 2, 3> {});
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
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, true /*skip_packer_strides*/, mutex_ADC>(
        pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

inline void _llk_pack_uninit_()
{
}

template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    namespace cfg = hal::cfg;

    // Select the Dest tile (channel-0 W).
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();

    // The previous tile must be fully written before the L1 destination changes.
    hal_llk::pack::wait_packer_idle_for_config();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(address);

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Rewind the face (Z) counters on both channels for the next tile.
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .apply();
}
