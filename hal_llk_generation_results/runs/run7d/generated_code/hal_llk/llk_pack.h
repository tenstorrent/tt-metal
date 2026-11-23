// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the packer LLK (row-major tile pack from Dest to L1).
//
// One PACR moves up to four consecutive Dest rows (one per Dest read interface). A face of face_r_dim rows
// therefore takes max(1, face_r_dim / 4) transfers; the last transfer of a face moves to the next face and
// the last transfer of the tile closes it in L1.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/utils/address_modifier.h"
#include "hal_llk_common.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

using namespace ckernel;

namespace hal_llk::pack_detail
{

// ADDR_MOD_PACK slots used by the tile pack sequence.
constexpr std::uint8_t ADDR_MOD_NEXT_ROWS = 0;
constexpr std::uint8_t ADDR_MOD_NEXT_FACE = 1;
constexpr std::uint8_t ADDR_MOD_TILE_DONE = 2;

// Dest rows moved by one PACR: one row per active Dest read interface.
inline std::uint32_t rows_per_transfer(const std::uint32_t face_r_dim)
{
    return face_r_dim < 4 ? face_r_dim : 4;
}

// Dest read interface mask for the given rows per transfer (zero selects all four interfaces).
inline std::uint8_t interface_mask(const std::uint32_t rows)
{
    return rows >= 4 ? 0 : static_cast<std::uint8_t>((1u << rows) - 1);
}

template <std::uint32_t RowStep>
inline void configure_addrmod()
{
    // Next transfer of the same face: advance the Dest row and the output row by the rows just moved.
    constexpr hal::pack::AddressModifier next_rows {
        .selection       = hal::cfg::Sec::S0,
        .source_row      = {.step = RowStep},
        .destination_row = {.step = RowStep},
    };
    // Face finished: restart rows, move to the next face.
    constexpr hal::pack::AddressModifier next_face {
        .selection        = hal::cfg::Sec::S1,
        .source_row       = {.clear = true},
        .destination_row  = {.clear = true},
        .source_face      = {.step = 1},
        .destination_face = {.step = 1},
    };
    // Tile finished: every counter back to the tile origin.
    constexpr hal::pack::AddressModifier tile_done {
        .selection        = hal::cfg::Sec::S2,
        .source_row       = {.clear = true},
        .destination_row  = {.clear = true},
        .source_face      = {.clear = true},
        .destination_face = {.clear = true},
    };
    hal::pack::configure_address_modifiers<next_rows, next_face, tile_done>();
}

inline void configure_addrmod(const std::uint32_t face_r_dim)
{
    switch (rows_per_transfer(face_r_dim))
    {
        case 1:
            configure_addrmod<1>();
            break;
        case 2:
            configure_addrmod<2>();
            break;
        default:
            configure_addrmod<4>();
            break;
    }
}

// Packer Dest addressing unit: the datum size of the packer input format (not the Dest storage width;
// a 16-bit packer input over a 32-bit Dest still steps in 16-bit units).
inline std::uint32_t dest_address_unit(const std::uint32_t pack_src_format)
{
    return register_datum_bytes(pack_src_format);
}

} // namespace hal_llk::pack_detail

/**
 * @brief Configure the packer address-modification (ADDR_MOD) slots for the selected pack mode.
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implementation supports PackMode::Default only");
    hal_llk::pack_detail::configure_addrmod(FACE_R_DIM);
}

/**
 * @brief Build and program the packer MOP template for the selected pack mode.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t num_tiles                   = 1)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implementation supports PackMode::Default only");
    using namespace hal_llk::pack_detail;

    const std::uint32_t rows                      = rows_per_transfer(face_r_dim);
    const std::uint32_t transfers_per_face        = face_r_dim / rows;
    constexpr hal::pack::DatumOverride datum_mode = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None;
    const std::uint8_t interfaces                 = interface_mask(rows);

    const hal::pack::DataTransfer next_rows {
        .address_modifier = ADDR_MOD_NEXT_ROWS,
        .interfaces       = interfaces,
        .datum_override   = datum_mode,
    };
    const hal::pack::DataTransfer next_face {
        .address_modifier = ADDR_MOD_NEXT_FACE,
        .interfaces       = interfaces,
        .datum_override   = datum_mode,
    };
    const hal::pack::DataTransfer tile_done {
        .address_modifier = ADDR_MOD_TILE_DONE,
        .interfaces       = interfaces,
        .datum_override   = datum_mode,
        .boundary         = hal::pack::TileBoundary::Last,
    };

    hal::mop::program(hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
        .outer_loop = {.count = num_faces * num_tiles},
        .inner_loop =
            {
                .count                               = transfers_per_face,
                .body_op                             = next_rows.get_operation(),
                .last_op_on_final_outer_iteration    = tile_done.get_operation(),
                .last_op_on_nonfinal_outer_iteration = next_face.get_operation(),
            },
    });
}

namespace llk_pack_internal_bh
{
/**
 * @brief Shared packer-init body behind the @ref _llk_pack_init_ overloads.
 */
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    [[maybe_unused]] const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implementation supports PackMode::Default only");
    static_assert(skip_packer_strides, "HAL pack init leaves the packer strides to _llk_pack_hw_configure_");
    static_assert(!mutex_ADC, "HAL pack init does not share the packer address counters");
    using namespace hal::sync;

    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
    if constexpr (!skip_addrmod_config)
    {
        hal_llk::pack_detail::configure_addrmod(face_r_dim);
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);

    // Every transfer reads one 16-datum row per interface (last datum on channel 0 X, first on channel 1 X,
    // the same assignment the unpacker needs).
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .X<FACE_C_DIM - 1>()
        .channel<hal::AddressChannel::Channel1>()
        .X<0>()
        .apply();
}
} // namespace llk_pack_internal_bh

/**
 * @brief One-time hardware configuration of the packer for a given data format and tile geometry.
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
    static_assert(pack_mode == PackMode::Default, "HAL pack implementation supports PackMode::Default only");
    namespace cfg = hal::cfg;
    using namespace hal::sync;
    using namespace hal_llk;

    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();

    // Dest addressing: one row, one face, one tile slot (64 rows).
    const std::uint32_t x_stride = pack_detail::dest_address_unit(pack_src_format);
    const std::uint32_t y_stride = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride = FACE_R_DIM * y_stride;
    const std::uint32_t w_stride = DEST_ROWS_PER_TILE * y_stride;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0>(x_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(y_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(w_stride),
        cfg::set<cfg::Packer[0].AddrBase[cfg::PackerReg::Reg0], cfg::Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrBase[cfg::PackerReg::Reg1], cfg::Sec::S0, 0>());

    // Packer register set 0: uncompressed output, formats, and the BFP exponent section (16B per face, or one for partial faces).
    const bool no_exponents          = is_format(pack_dst_format, DataFormat::Lf8) || is_format(pack_dst_format, DataFormat::Int8);
    const std::uint32_t exp_sections = no_exponents ? 0 : (partial_face ? 1 : num_faces);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_sections),
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
        cfg::set<cfg::Thcon[cfg::Reg1].L1_source_addr, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0, 0>());

    // How the packer reads Dest: 32-bit storage whenever Dest accumulates in FP32 or holds 32-bit data.
    const bool dest_32b      = is_fp32_dest_acc_en || x_stride == 4;
    const bool unsigned_int8 = is_format(pack_src_format, DataFormat::UInt8);
    const bool int8_src      = is_format(pack_src_format, DataFormat::Int8) || unsigned_int8;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(dest_32b ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(unsigned_int8 ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>((int8_src && !dest_32b) ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, 0>());

    // One Dest read per XY plane on every packer; no edge masking.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S1, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S1, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S1, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S1, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S1, 0>(),
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S2, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S2, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S2, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S2, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S2, 0>(),
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S3, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S3, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S3, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S3, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S3, 0>(),
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());

    // ReLU applied on the Dest read path: mode in the low nibble, threshold in the upper half-word.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0xF), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xFFFF));
}

/**
 * @brief Initialize the packer (addrmod + MOP + strides) for a pack op given a source format.
 */
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
    // Strides are owned by _llk_pack_hw_configure_; init only programs the transfer sequence.
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, true, mutex_ADC>(
        pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

/**
 * @brief Tear down the packer after a pack op (no-op on Blackhole).
 */
inline void _llk_pack_uninit_()
{
}

/**
 * @brief Pack one tile from the destination register to an L1 address.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "HAL pack implementation supports PackMode::Default only");
    static_assert(!mutex_ADC, "HAL pack does not share the packer address counters");
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    // Select the Dest tile slot to read.
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();

    // The L1 destination may only change once the previous tile has been written.
    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(address);

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    hal_llk::pack_detail::reset_packer_counters();
}
