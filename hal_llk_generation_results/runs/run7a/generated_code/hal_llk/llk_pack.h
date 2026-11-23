// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole pack LLK functions.

#pragma once

#include <cstdint>
#include <utility>

#include "hal/pack.h"
#include "hal/utils/address_modifier.h"
#include "llk_hal_common.h"
#include "llk_pack_common.h"

namespace hal_llk::pack
{

// Packer ADDR_MOD_PACK slots.
constexpr std::uint8_t ADDR_MOD_NEXT_ROW  = 0;
constexpr std::uint8_t ADDR_MOD_NEXT_FACE = 1;
constexpr std::uint8_t ADDR_MOD_TILE_END  = 2;

template <std::size_t... Rows>
inline void map_all_rows_to_edge_mask_0(std::index_sequence<Rows...>)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::TileRowSetMapping[0][Rows], cfg::Sec::S0, 0>()...);
}

// Dest read strides of the packer: one row per Y step, one face per Z step, one 32x32 tile per W step.
inline void set_packer_strides(const std::uint32_t pack_src_format)
{
    const std::uint32_t x_stride = register_datum_bytes(pack_src_format);
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride = FACE_R_DIM * y_stride;
    const std::uint32_t w_stride = TILE_NUM_FACES * z_stride;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(y_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(w_stride));
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces, const bool partial_face)
{
    const std::uint32_t dst = pack_dst_format & 0x1F;
    const std::uint32_t src = pack_src_format & 0x1F;
    // Exponent section (16B per 16 rows) is used by BFP outputs; Lf8/Int8 have none.
    const std::uint32_t exp_section_size =
        ((dst == fmt(DataFormat::Lf8)) || (dst == fmt(DataFormat::Int8))) ? 0 : (partial_face ? 1 : num_faces);
    const bool is_32b_src = (src == fmt(DataFormat::Float32)) || (src == fmt(DataFormat::Int32)) || (src == fmt(DataFormat::UInt32));

    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_section_size),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_zero_compress, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].ovrd_default_throttle_mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Out_data_format, cfg::Sec::S0>(pack_dst_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg1].In_data_format, cfg::Sec::S0>(pack_src_format & DATA_FORMAT_CONFIG_MASK),
        cfg::set<cfg::Thcon[cfg::Reg1].Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Source_interface_selection, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].All_pack_disable_zero_compress_ovrd, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].L1_source_addr, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0>(dst == fmt(DataFormat::Fp8_e4m3)),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0, 0>(),
        // Dest read mode
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(is_fp32_dest_acc_en || is_32b_src),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(src == fmt(DataFormat::UInt8)),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>(!(is_fp32_dest_acc_en || is_32b_src) && ((src & 0xF) == fmt(DataFormat::Int8))),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0>(dst == fmt(DataFormat::Float16)));
}

// One PACR per Dest row; the last row of a face steps to the next face, the last row of the tile closes it.
template <bool zero_output>
inline void configure_mop(const std::uint32_t face_r_dim, const std::uint32_t num_faces, const std::uint32_t num_tiles)
{
    constexpr auto datum_override = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None;
    // A single Dest-read interface reads the one row of each PACR (the all-interfaces mask reads several rows).
    constexpr std::uint8_t interfaces = 0b0001;
    constexpr std::uint32_t pack_row =
        hal::pack::DataTransfer {.address_modifier = ADDR_MOD_NEXT_ROW, .interfaces = interfaces, .datum_override = datum_override}.get_operation();
    constexpr std::uint32_t pack_face_end =
        hal::pack::DataTransfer {.address_modifier = ADDR_MOD_NEXT_FACE, .interfaces = interfaces, .datum_override = datum_override}.get_operation();
    constexpr std::uint32_t pack_tile_end =
        hal::pack::DataTransfer {
            .address_modifier = ADDR_MOD_TILE_END, .interfaces = interfaces, .datum_override = datum_override, .boundary = hal::pack::TileBoundary::Last}
            .get_operation();

    LLK_ASSERT(num_tiles == 1, "HAL pack supports one tile per MOP run");
    hal::mop::program(hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
        .outer_loop = {.count = num_faces},
        .inner_loop =
            {.count                               = face_r_dim,
             .body_op                             = pack_row,
             .last_op_on_final_outer_iteration    = pack_tile_end,
             .last_op_on_nonfinal_outer_iteration = pack_face_end},
    });
}

} // namespace hal_llk::pack

template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");
    using hal::pack::AddressModifier;
    constexpr AddressModifier next_row {.selection = hal::cfg::Sec::S0, .source_row = {.step = 1}};
    constexpr AddressModifier next_face {.selection = hal::cfg::Sec::S1, .source_row = {.clear = true}, .source_face = {.step = 1}};
    constexpr AddressModifier tile_end {.selection = hal::cfg::Sec::S2, .source_row = {.clear = true}, .source_face = {.clear = true}};
    hal::pack::configure_address_modifiers<next_row, next_face, tile_end>();
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const std::uint32_t num_tiles  = 1)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");
    hal_llk::pack::configure_mop<zero_output>(face_r_dim, num_faces, num_tiles);
}

namespace llk_pack_internal_bh
{
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles)
{
    static_assert(!mutex_ADC, "HAL pack does not borrow ADCs across threads");
    using hal_llk::pack::AdcChannel;
    using hal_llk::pack::AdcClient;

    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        hal_llk::sync::wait::stall<hal_llk::sync::StallTarget::Config, hal_llk::sync::StallCondition::PackerIdle>();
        hal_llk::pack::set_packer_strides(pack_src_format);
    }
    // Each PACR reads one row of FACE_C_DIM datums: last datum index in channel-0 X, first datum in channel-1 X.
    hal::address_counters.client<AdcClient::Packers>().channel<AdcChannel::Channel0>().X<FACE_C_DIM - 1>().channel<AdcChannel::Channel1>().X<0>().apply();
}
} // namespace llk_pack_internal_bh

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false)
{
    hal_llk::sync::wait::stall<hal_llk::sync::StallTarget::Config, hal_llk::sync::StallCondition::PackerIdle>();
    hal_llk::pack::set_packer_strides(pack_src_format);
    hal_llk::pack::set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);
}

/**
 * @brief Configure the packer formats, Dest strides, ReLU, edge masks and counters.
 */
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
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    hal_llk::pack::set_packer_strides(pack_src_format);
    hal_llk::pack::set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    const ReluConfig relu = ReluConfig::from_packed(relu_config);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(static_cast<std::uint32_t>(relu.get_mode())),
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(relu.get_threshold()),
        // No edge masking: every row uses edge-offset set 0 with all datums enabled.
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>(),
        // One Dest read per XY plane.
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());
    hal_llk::pack::map_all_rows_to_edge_mask_0(std::make_index_sequence<16> {});
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
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, mutex_ADC>(
        pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

inline void _llk_pack_uninit_()
{
}

/**
 * @brief Pack one tile from Dest to L1.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implements PackMode::Default only");
    static_assert(!mutex_ADC, "HAL pack does not borrow ADCs across threads");
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;
    using hal_llk::pack::AdcChannel;
    using hal_llk::pack::AdcClient;

    // Select the Dest tile.
    hal::runtime_address_counters.client<AdcClient::Packers>().channel<AdcChannel::Channel0>().W(tile_index).apply();

    // The previous tile must be written out before the L1 destination changes.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(address);

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
