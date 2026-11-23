// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the pack LLK (Blackhole), PackMode::Default.
//
// Model: one PACR moves one 16-datum Dest row of the selected face to L1. The MOP packs one tile:
// face_r_dim rows per face (row step), then the next face (face step), and the final row closes the tile.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/gpr_ops.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/utils/address_modifier.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_hal_common.h"
#include "llk_pack_common.h"

namespace hal_llk::pack_detail
{

constexpr std::uint8_t ADDR_MOD_ROW  = 0; // next Dest row
constexpr std::uint8_t ADDR_MOD_FACE = 1; // first row of the next face
constexpr std::uint8_t ADDR_MOD_TILE = 2; // back to row 0 of face 0

template <std::uint8_t addr_mod, hal::pack::TileBoundary boundary, bool zero_output>
constexpr std::uint32_t row_transfer_op()
{
    return hal::pack::DataTransfer {
        .address_modifier = addr_mod,
        .interfaces       = 0b0001, // a single Dest-read interface moves one 16-datum row
        .datum_override   = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None,
        .boundary         = boundary,
    }
        .get_operation();
}

// Packer Dest-side strides: one row, one 16-row face, one 64-row tile slot (bytes of the Dest datum format).
inline void set_packer_strides(const std::uint32_t pack_src_format)
{
    namespace cfg          = hal::cfg;
    const std::uint32_t x  = register_datum_bytes(pack_src_format);
    const std::uint32_t ys = FACE_C_DIM * x;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(ys),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(FACE_R_DIM * ys),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(TILE_NUM_FACES * FACE_R_DIM * ys));
}

} // namespace hal_llk::pack_detail

/**
 * @brief Configure the packer address-modification (ADDR_MOD) slots for the selected pack mode.
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "only PackMode::Default is implemented");
    using hal::pack::AddressModifier;
    constexpr AddressModifier next_row {
        .selection       = hal::cfg::Sec::S0,
        .source_row      = {.step = 1},
        .destination_row = {.step = 1},
    };
    constexpr AddressModifier next_face {
        .selection        = hal::cfg::Sec::S1,
        .source_row       = {.clear = true},
        .destination_row  = {.clear = true},
        .source_face      = {.step = 1},
        .destination_face = {.step = 1},
    };
    constexpr AddressModifier restart_tile {
        .selection        = hal::cfg::Sec::S2,
        .source_row       = {.clear = true},
        .destination_row  = {.clear = true},
        .source_face      = {.clear = true},
        .destination_face = {.clear = true},
    };
    hal::pack::configure_address_modifiers<next_row, next_face, restart_tile>();
}

/**
 * @brief Build and program the packer MOP template for the selected pack mode.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, [[maybe_unused]] const std::uint32_t num_tiles = 1)
{
    static_assert(pack_mode == PackMode::Default, "only PackMode::Default is implemented");
    using namespace hal_llk::pack_detail;
    using hal::pack::TileBoundary;

    hal::mop::MopConfig<hal::mop::MopTemplate::Template1> mop {};
    mop.outer_loop.count                               = num_faces;
    mop.inner_loop.count                               = face_r_dim;
    mop.inner_loop.body_op                             = row_transfer_op<ADDR_MOD_ROW, TileBoundary::NotLast, zero_output>();
    mop.inner_loop.last_op_on_nonfinal_outer_iteration = row_transfer_op<ADDR_MOD_FACE, TileBoundary::NotLast, zero_output>();
    mop.inner_loop.last_op_on_final_outer_iteration    = row_transfer_op<ADDR_MOD_TILE, TileBoundary::Last, zero_output>();
    hal::mop::program(mop);
}

namespace llk_pack_internal_bh
{
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    const std::uint32_t pack_src_format, const std::uint32_t face_r_dim, const std::uint32_t tile_c_dim, const std::uint32_t num_faces, const std::uint32_t num_tiles)
{
    static_assert(!mutex_ADC, "mutex_ADC is not implemented");
    namespace sync = hal::sync;

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        hal_llk::pack_detail::set_packer_strides(pack_src_format);
    }

    // Each PACR reads one row: X runs 0..15 (end on channel 0, start on channel 1; see the unpack datum count).
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
    static_assert(pack_mode == PackMode::Default, "only PackMode::Default is implemented");
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    using namespace hal_llk;

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    pack_detail::set_packer_strides(pack_src_format);

    // ReLU lives in an ALU word shared with the math thread.
    sync::mutex::acquire<sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0xF), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xFFFF));
    sync::mutex::release<sync::Mutex::M0>();

    // Packer configuration: formats, uncompressed output, exponent section (16-byte units, one per full face).
    // Only block-float outputs carry an exponent section, so other formats program a zero-sized one.
    const std::uint32_t dst_code = pack_dst_format & 0x1F;
    const std::uint32_t exp_size = is_bfp_format(pack_dst_format) ? (partial_face ? 1 : num_faces) : 0;
    const bool exp_threshold     = is_fp32_dest_acc_en && is_bfp_a_format(pack_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_size),
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
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0>(exp_threshold ? 1 : 0),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold, cfg::Sec::S0>(exp_threshold ? 113 : 0),
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0>(dst_code == fmt(DataFormat::Fp8_e4m3) ? 1 : 0));

    // Dest read mode: 32-bit Dest when FP32 accumulation or a 32-bit source format is used.
    const bool read_32b   = is_fp32_dest_acc_en || is_32bit_format(pack_src_format);
    const bool round_10b  = is_fp32_dest_acc_en && (pack_src_format != fmt(DataFormat::Float32)) && !is_integer_format(pack_src_format);
    const bool read_uint8 = pack_src_format == fmt(DataFormat::UInt8);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(read_32b ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(read_uint8 ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0>(round_10b ? 1 : 0));

    // Pass every datum: one edge mask (all ones) selected for every row, no per-face mask sets.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>());
    cfg::TileRowSetMapping[0].forEach([&](auto S) { cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[0][S], cfg::Sec::S0, 0>(); });

    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());
}

/**
 * @brief Initialize the packer (addrmod + MOP + strides) for a pack op given a source format.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false, bool mutex_ADC = false>
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
 * @brief Pack one tile from the destination register to an L1 address.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "only PackMode::Default is implemented");
    static_assert(!mutex_ADC, "mutex_ADC is not implemented");
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // The previous tile must be closed before its Dest slot and L1 destination are replaced.
    sync::wait::stall<sync::StallTarget::Config | sync::StallTarget::Thcon, sync::StallCondition::PackerIdle>();

    // Select the Dest tile slot (channel-0 W).
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();

    // The L1 destination is replaced as one complete word from a GPR (WRCFG). The packer only picks up a new
    // destination at a write-aligner flush (otherwise at the next tile close, i.e. one tile late), so once the
    // write has landed a no-data flush latches it. The flush PACR applies ADDR_MOD 0, so the row/face
    // positions are restarted afterwards.
    constexpr auto output_addr = hal::gpr<ckernel::p_gpr_pack::OUTPUT_ADDR>();
    hal::gpr_ops::set(output_addr, address);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(output_addr);
    sync::wait::stall<sync::StallTarget::Pack, sync::StallCondition::ConfigUnitIdle>();
    hal::pack::flush_write_aligners();
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Y<0>()
        .Z<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Y<0>()
        .Z<0>()
        .apply();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
