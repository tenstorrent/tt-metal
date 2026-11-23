// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "hal/gpr_ops.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/utils/address_modifier.h"
#include "llk_hal_common.h"
#include "llk_pack_common.h"

namespace hal_llk
{

// Packer ADDR_MOD slots.
inline constexpr std::uint8_t PACK_AM_ROWS     = 0; // next four dest rows
inline constexpr std::uint8_t PACK_AM_FACE_END = 1; // back to row 0 of the next face
inline constexpr std::uint8_t PACK_AM_TILE_END = 2; // back to row 0 of face 0

template <std::size_t M, std::size_t... S>
inline void clear_tile_row_set_mapping(std::index_sequence<S...>)
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::TileRowSetMapping[M][S], cfg::Sec::S0, 0>()...);
}

// Dest-side (channel 0) strides in bytes of the packer input format: row, face and 32x32 tile slot.
inline void set_packer_strides(const std::uint32_t pack_src_format)
{
    namespace cfg                   = hal::cfg;
    constexpr auto Ch0              = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0];
    const std::uint32_t datum       = datum_bytes(pack_src_format);
    const std::uint32_t row_bytes   = FACE_C_DIM * datum;
    const std::uint32_t face_bytes  = FACE_R_DIM * row_bytes;
    const std::uint32_t tile_bytes  = TILE_NUM_FACES * face_bytes;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<Ch0.Xstride, cfg::Sec::S0, 0>(),
        cfg::set<Ch0.Ystride, cfg::Sec::S0>(row_bytes),
        cfg::set<Ch0.Zstride, cfg::Sec::S0>(face_bytes),
        cfg::set<Ch0.Wstride, cfg::Sec::S0>(tile_bytes));
}

} // namespace hal_llk

template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    using hal::pack::AddressModifier;
    hal::pack::configure_address_modifiers<
        AddressModifier {.selection = hal::cfg::Sec::S0, .source_row = {.step = 4}, .destination_row = {.step = 4}},
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

/**
 * @brief Program the pack MOP: per face, face_r_dim rows read four at a time through the dest-read
 *        interfaces; the final transfer closes the tile.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const std::uint32_t num_tiles = 1)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    using namespace hal::mop;

    // One dest-read interface per row: 1, 2 or all 4 rows per transfer.
    const std::uint8_t interfaces = face_r_dim == 1 ? 0b0001 : (face_r_dim == 2 ? 0b0011 : 0b0000);
    const std::uint32_t row_ops   = face_r_dim < 4 ? 1 : face_r_dim / 4;

    hal::pack::DataTransfer transfer {
        .address_modifier = hal_llk::PACK_AM_ROWS,
        .interfaces       = interfaces,
        .datum_override   = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None,
    };
    const std::uint32_t rows = transfer.get_operation();

    transfer.address_modifier    = hal_llk::PACK_AM_FACE_END;
    const std::uint32_t face_end = transfer.get_operation();

    transfer.address_modifier    = hal_llk::PACK_AM_TILE_END;
    transfer.boundary            = hal::pack::TileBoundary::Last;
    const std::uint32_t tile_end = transfer.get_operation();

    program(MopConfig<MopTemplate::Template1> {
        .outer_loop = {.count = num_faces * num_tiles},
        .inner_loop =
            {.count                               = row_ops,
             .body_op                             = rows,
             .last_op_on_final_outer_iteration    = tile_end,
             .last_op_on_nonfinal_outer_iteration = face_end},
    });
}

/**
 * @brief One-time packer configuration: formats, strides, dest read mode, masks, counters, ReLU.
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
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    sync::wait::stall<sync::StallTarget::Config | sync::StallTarget::Thcon, sync::StallCondition::PackerIdle>();

    hal_llk::set_packer_strides(pack_src_format);

    // Packer register set 0: formats, uncompressed output read from dest, no L1 accumulation.
    const std::uint32_t dst_hw        = hal_llk::hw_format(pack_dst_format);
    const bool no_exponent_section    = (dst_hw == hal_llk::hw_format(static_cast<std::uint32_t>(DataFormat::Lf8))) ||
                                     (dst_hw == hal_llk::hw_format(static_cast<std::uint32_t>(DataFormat::Int8)));
    const std::uint32_t exp_sec_size  = no_exponent_section ? 0 : (partial_face ? 1 : num_faces);
    constexpr auto R1                 = cfg::Thcon[cfg::Reg1];
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R1.Row_start_section_size, cfg::Sec::S0, 0>(), cfg::set<R1.Exp_section_size, cfg::Sec::S0>(exp_sec_size));
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<R1.Disable_zero_compress, cfg::Sec::S0, 1>(),
        cfg::set<R1.Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        cfg::set<R1.Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<R1.ovrd_default_throttle_mode, cfg::Sec::S0, 0>(),
        cfg::set<R1.Out_data_format, cfg::Sec::S0>(dst_hw),
        cfg::set<R1.In_data_format, cfg::Sec::S0>(hal_llk::hw_format(pack_src_format)),
        cfg::set<R1.Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<R1.Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<R1.Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<R1.Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<R1.Source_interface_selection, cfg::Sec::S0, 0>(),
        cfg::set<R1.pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<R1.All_pack_disable_zero_compress_ovrd, cfg::Sec::S0, 0>(),
        cfg::set<R1.Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<R1.pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>(),
        cfg::set<R1.L1_source_addr, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<R1.Pack_L1_Acc, cfg::Sec::S0, 0>(), cfg::set<R1.Exp_threshold_en, cfg::Sec::S0, 0>(), cfg::set<R1.Pac_LF8_4b_exp, cfg::Sec::S0, 0>());

    // How the packer reads dest.
    const bool src_32b = is_fp32_dest_acc_en || hal_llk::is_32bit_format(pack_src_format);
    const bool src_int8 =
        pack_src_format == static_cast<std::uint32_t>(DataFormat::Int8) || pack_src_format == static_cast<std::uint32_t>(DataFormat::UInt8);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(src_32b ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(pack_src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>((!src_32b && src_int8) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, 0>());

    // Every row passes the edge mask: all rows map to mask set 0, which keeps all 16 datums.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(), cfg::set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>());
    hal_llk::clear_tile_row_set_mapping<0>(std::make_index_sequence<16> {});
    hal_llk::clear_tile_row_set_mapping<1>(std::make_index_sequence<16> {});
    hal_llk::clear_tile_row_set_mapping<2>(std::make_index_sequence<16> {});
    hal_llk::clear_tile_row_set_mapping<3>(std::make_index_sequence<16> {});

    // Packer counters: one dest read per XY plane, no automatic last/context handling.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());

    // ReLU: mode in bits [1:0], threshold in bits [31:16] of relu_config.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0x3), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xFFFF));
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
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    static_assert(!mutex_ADC, "mutex_ADC is not implemented");
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        namespace sync = hal::sync;
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
        hal_llk::set_packer_strides(pack_src_format);
    }

    // Each dest-read interface moves one 16-datum row (X 0..15). The two channels are written
    // separately: the combined X/X form of the builder (SETADCXX) did not program start/end as
    // expected on hardware (the unpacker hung with it).
    hal::address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().X<0>().apply();
    hal::address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel1>().X<FACE_C_DIM - 1>().apply();
}

inline void _llk_pack_uninit_()
{
}

/**
 * @brief Pack dest tile tile_index to L1 address `address`.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    static_assert(!mutex_ADC, "mutex_ADC is not implemented");
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // L1 destination: GPR load, then one WRCFG once the previous tile has left the packer.
    hal::gpr_ops::set(hal::gpr<hal_llk::PACK_GPR_OUTPUT_ADDR>(), address);
    sync::wait::stall<sync::StallTarget::Thcon | sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<hal_llk::PACK_GPR_OUTPUT_ADDR>());

    // Source tile in dest (W counter); also separates the WRCFG from the first PACR.
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
