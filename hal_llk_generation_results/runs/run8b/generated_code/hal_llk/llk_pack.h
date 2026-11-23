// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL implementation of the packer (Dest -> L1) for Blackhole, PackMode::Default.

#pragma once

#include <cstdint>

#include "llk_hal_common.h"
#include "llk_pack_common.h"

using namespace ckernel;

namespace hal_llk
{

// Packer address-modifier slots.
constexpr std::uint8_t PACK_ADDR_MOD_ROW       = 0; // next row of the face
constexpr std::uint8_t PACK_ADDR_MOD_FACE_DONE = 1; // first row of the next face
constexpr std::uint8_t PACK_ADDR_MOD_TILE_DONE = 2; // first row of the first face

// Bit 31 of the packer L1 destination address marks a newly programmed tile address.
constexpr std::uint32_t PACK_L1_DEST_ADDR_VALID = 1u << 31;

// Packer channel-0 strides over Dest, in bytes of the Dest datum: one row, one face, one tile slot.
inline void packer_configure_strides(const std::uint32_t pack_src_format)
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

// One PACR moves one 16-datum row through Dest-read interface 0.
constexpr hal::pack::DataTransfer pack_row(const std::uint8_t address_modifier, const bool last, const bool zero_output)
{
    return {
        .address_modifier = address_modifier,
        .interfaces       = 0b0001,
        .datum_override   = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None,
        .boundary         = last ? hal::pack::TileBoundary::Last : hal::pack::TileBoundary::NotLast,
    };
}

} // namespace hal_llk

template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "HAL pack supports PackMode::Default only");
    using hal::pack::AddressModifier;
    hal::pack::configure_address_modifiers<
        AddressModifier {.selection = hal::cfg::Sec::S0, .source_row = {.step = 1}, .destination_row = {.step = 1}},
        AddressModifier {
            .selection        = hal::cfg::Sec::S1,
            .source_row       = {.clear = true},
            .destination_row  = {.clear = true},
            .source_face      = {.step = 1},
            .destination_face = {.step = 1}},
        AddressModifier {
            .selection        = hal::cfg::Sec::S2,
            .source_row       = {.clear = true},
            .destination_row  = {.clear = true},
            .source_face      = {.clear = true},
            .destination_face = {.clear = true}}>();
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t num_tiles                   = 1)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack supports PackMode::Default only");
    using namespace hal_llk;
    constexpr hal::mop::MopConfig<hal::mop::MopTemplate::Template1> pack_tile {
        .outer_loop = {.count = 0},
        .inner_loop =
            {
                .count                               = 0,
                .body_op                             = pack_row(PACK_ADDR_MOD_ROW, false, zero_output).get_operation(),
                .last_op_on_final_outer_iteration    = pack_row(PACK_ADDR_MOD_TILE_DONE, true, zero_output).get_operation(),
                .last_op_on_nonfinal_outer_iteration = pack_row(PACK_ADDR_MOD_FACE_DONE, false, zero_output).get_operation(),
            },
    };
    // Outer loop: faces, inner loop: rows of a face.
    hal::mop::program<pack_tile>(
        hal::mop::runtime_field<hal::mop::Template1Field::OuterLoopCount>(num_faces * num_tiles),
        hal::mop::runtime_field<hal::mop::Template1Field::InnerLoopCount>(face_r_dim));
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
    static_assert(!mutex_ADC, "HAL pack does not serialize ADC access across threads");
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
        hal_llk::packer_configure_strides(pack_src_format);
    }
    // Every PACR reads one row of FACE_C_DIM datums (end on Channel0 X, start on Channel1 X).
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
    static_assert(pack_mode == PackMode::Default, "HAL pack supports PackMode::Default only");
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    using namespace hal_llk;

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    packer_configure_strides(pack_src_format);

    // Exponent section (one 16-byte line per face, or one for partial faces); none for formats without one.
    const std::uint32_t dst_code     = pack_dst_format & 0x1f;
    const bool no_exp_section        = dst_code == static_cast<std::uint32_t>(DataFormat::Lf8) || dst_code == static_cast<std::uint32_t>(DataFormat::Int8);
    const std::uint32_t exp_sections = no_exp_section ? 0 : (partial_face ? 1 : num_faces);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_sections));
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_zero_compress, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].ovrd_default_throttle_mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Out_data_format, cfg::Sec::S0>(format_code(pack_dst_format)),
        cfg::set<cfg::Thcon[cfg::Reg1].In_data_format, cfg::Sec::S0>(format_code(pack_src_format)),
        cfg::set<cfg::Thcon[cfg::Reg1].Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Source_interface_selection, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].All_pack_disable_zero_compress_ovrd, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].L1_source_addr, cfg::Sec::S0, 0>());
    const bool lf8_out = dst_code == static_cast<std::uint32_t>(DataFormat::Lf8);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0>(lf8_out ? 1 : 0));

    // How the packer reads Dest.
    const std::uint32_t src_code = pack_src_format & 0x1f;
    const bool read_32b          = is_fp32_dest_acc_en || is_32bit_register_format(pack_src_format);
    const bool round_10b         = is_fp32_dest_acc_en && src_code != static_cast<std::uint32_t>(DataFormat::Float32) && !is_32bit_register_format(pack_src_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(read_32b ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(is_uint8_format(pack_src_format) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>((!read_32b && is_int8_math_format(pack_src_format)) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0>(round_10b ? 1 : 0));

    // Every datum of every row passes the edge mask.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xffff>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(), cfg::set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {0});

    // Default (non-reduce) packer counters.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());

    // ReLU applied on the way out of Dest.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0x3), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xffff));
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
 * @brief Pack Dest tile `tile_index` of the current section to L1 address `address`.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack supports PackMode::Default only");
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // PACK_TILE_SEM is held from the address store until the packer has drained the tile, so the
    // MMIO store never changes an L1 destination the packer may still be using.
    while (sync::semaphore::read<hal_llk::PACK_TILE_SEM>() > 0)
    {
    }
    const std::uint32_t l1_dest = address | hal_llk::PACK_L1_DEST_ADDR_VALID;
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(l1_dest);
    // The PACRs below must not be issued before the store has reached the CFG register.
    while (cfg::read<cfg::Access::MMIO, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>() != l1_dest)
    {
    }
    sync::semaphore::post<sync::Access::MMIO, hal_llk::PACK_TILE_SEM>();

    set_dst_write_addr<mutex_ADC>(tile_index);
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Release the destination once this tile has left the packer.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::PackerIdle>();
    sync::semaphore::get<hal_llk::PACK_TILE_SEM>();
}
