// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the tile packer (Blackhole, PackMode::Default).

#pragma once

#include <array>
#include <cstdint>

#include "hal/gpr_ops.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/utils/address_modifier.h"
#include "hal_llk_common.h"
#include "llk_memory_checks.h"
#include "llk_pack_common.h"

namespace hal_llk
{

// GPR of the pack thread that carries the per-tile L1 destination address.
inline constexpr std::uint32_t PACK_OUTPUT_ADDR_GPR = p_gpr_pack::OUTPUT_ADDR;

// Faces per tile of the programmed pack MOP (rewritten by _llk_pack_ to wait for the previous expansion).
inline std::uint32_t pack_faces_per_tile = 4;

// Start a tile: rewind Y and Z and select Dest tile `tile_index` (W). Both channels are written; the L1-side
// (channel-1) strides are zero, so only the Dest-side values affect addressing.
inline void start_packer_tile(const std::uint32_t tile_index)
{
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Y<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Y<0>()
        .apply();
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Z(0)
        .W(tile_index)
        .channel<hal::AddressChannel::Channel1>()
        .Z(0)
        .W(tile_index)
        .apply();
}

// Packer address-modifier slots.
inline constexpr std::uint8_t PACK_AM_ROWS = 0; // next group of Dest rows
inline constexpr std::uint8_t PACK_AM_FACE = 2; // first row of the next face

// Dest rows moved by one PACR: one row per Dest-read interface, up to all four.
inline std::uint32_t pack_rows_per_transfer(const std::uint32_t face_r_dim)
{
    return face_r_dim >= 4 ? 4 : face_r_dim;
}

template <std::uint32_t Rows>
inline void program_pack_addrmods()
{
    constexpr hal::pack::AddressModifier next_rows {
        .selection       = hal::cfg::Sec::S0,
        .source_row      = {.step = Rows},
        .destination_row = {.step = Rows},
    };
    constexpr hal::pack::AddressModifier unused {
        .selection = hal::cfg::Sec::S1,
    };
    constexpr hal::pack::AddressModifier next_face {
        .selection       = hal::cfg::Sec::S2,
        .source_row      = {.clear = true},
        .destination_row = {.step = Rows},
        .source_face     = {.step = 1},
    };
    constexpr hal::pack::AddressModifier unused3 {
        .selection = hal::cfg::Sec::S3,
    };
    hal::pack::configure_address_modifiers<next_rows, unused, next_face, unused3>();
}

// Dest-side strides of the packer (channel 0): row, face and tile; the datum (X) step is implicit.
inline void program_packer_strides(const std::uint32_t pack_src_format)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    const std::uint32_t y_stride = FACE_C_DIM * register_datum_bytes(pack_src_format);
    const std::uint32_t z_stride = FACE_R_DIM * y_stride;
    const std::uint32_t w_stride = TILE_NUM_FACES * z_stride;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0, 0>(),
        set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(y_stride),
        set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(z_stride),
        set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(w_stride));
}

// Packer register set 0: formats, exponent section of BFP outputs, uncompressed output.
template <bool is_fp32_dest_acc_en>
inline void program_packer_config(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces, const bool partial_face)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    // Exponent section (16 B per face, one for partial faces); integer outputs carry none.
    const bool integer_output            = is_int8_math_format(pack_dst_format);
    const std::uint32_t exp_section_size = integer_output ? 0 : (partial_face ? 1 : num_faces);
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_section_size),
        set<cfg::Thcon[cfg::Reg1].Disable_zero_compress, cfg::Sec::S0, 1>(),
        set<cfg::Thcon[cfg::Reg1].Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].ovrd_default_throttle_mode, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Out_data_format, cfg::Sec::S0>(format_code(pack_dst_format)),
        set<cfg::Thcon[cfg::Reg1].In_data_format, cfg::Sec::S0>(format_code(pack_src_format)),
        set<cfg::Thcon[cfg::Reg1].Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Enable_out_fifo, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Source_interface_selection, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].pack_start_intf_pos, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].All_pack_disable_zero_compress_ovrd, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Add_tile_header_size, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].L1_source_addr, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0, 0>(),
        set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0, 0>());

    // Dest read mode: 32-bit Dest datums, or Int8 datums in a 16-bit Dest.
    const bool read_32b = is_fp32_dest_acc_en || is_32bit_register_format(pack_src_format);
    const bool is_8bit_int =
        (pack_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Int8) || (pack_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::UInt8);
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(read_32b ? 1 : 0),
        set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>((pack_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::UInt8) ? 1 : 0),
        set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>((!read_32b && is_8bit_int) ? 1 : 0),
        set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, 0>());
}

// Pass-through edge masks and one Dest read per XY plane.
inline void program_packer_counters_and_masks()
{
    namespace cfg = hal::cfg;
    using cfg::set;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(), set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>());
    // Every row of every face uses edge-offset section 0 (the all-ones mask).
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {0});
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[1][0], cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {0});
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[2][0], cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {0});
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[3][0], cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {0});
}

template <bool zero_output>
inline std::uint32_t pack_row_operation(const std::uint8_t addr_mod, const std::uint32_t rows, const hal::pack::TileBoundary boundary)
{
    return hal::pack::DataTransfer {
        .address_modifier = addr_mod,
        .interfaces       = static_cast<std::uint8_t>((1u << rows) - 1),
        .datum_override   = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None,
        .boundary         = boundary,
    }
        .get_operation();
}

} // namespace hal_llk

namespace hal_llk
{
// Address modifiers for faces of face_r_dim rows (row-group step = rows moved by one PACR).
inline void program_pack_addrmods_for_face(const std::uint32_t face_r_dim)
{
    switch (pack_rows_per_transfer(face_r_dim))
    {
        case 1:
            program_pack_addrmods<1>();
            break;
        case 2:
            program_pack_addrmods<2>();
            break;
        default:
            program_pack_addrmods<4>();
            break;
    }
}
} // namespace hal_llk

/**
 * @brief Configure the packer address modifiers: next group of 4 rows (slot 0), next face (slot 2).
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    hal_llk::program_pack_addrmods_for_face(FACE_R_DIM);
}

/**
 * @brief Program the packer MOP: faces in the outer loop, groups of rows (one per Dest-read interface) in the
 * inner loop; the last PACR of a face steps to the next face and the last PACR of the tile closes it.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    [[maybe_unused]] const std::uint32_t num_tiles = 1)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    using hal::pack::TileBoundary;
    const std::uint32_t rows = hal_llk::pack_rows_per_transfer(face_r_dim);
    hal::mop::MopConfig<hal::mop::MopTemplate::Template1> config {};
    config.outer_loop.count                               = num_faces;
    config.inner_loop.count                               = face_r_dim / rows;
    config.inner_loop.body_op                             = hal_llk::pack_row_operation<zero_output>(hal_llk::PACK_AM_ROWS, rows, TileBoundary::NotLast);
    config.inner_loop.last_op_on_nonfinal_outer_iteration = hal_llk::pack_row_operation<zero_output>(hal_llk::PACK_AM_FACE, rows, TileBoundary::NotLast);
    config.inner_loop.last_op_on_final_outer_iteration    = hal_llk::pack_row_operation<zero_output>(hal_llk::PACK_AM_FACE, rows, TileBoundary::Last);
    hal::mop::program(config);
    hal_llk::pack_faces_per_tile = num_faces;
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
    static_assert(!mutex_ADC, "mutex_ADC is not supported by the HAL implementation");
    using namespace hal::sync;

    wait::stall<StallTarget::All, StallCondition::PackerIdle>();
    if constexpr (!skip_addrmod_config)
    {
        hal_llk::program_pack_addrmods_for_face(face_r_dim);
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        hal_llk::program_packer_strides(pack_src_format);
    }
    // Each Dest-read interface moves one 16-datum row per PACR: SETADCXX with X end FACE_C_DIM - 1 and X start 0
    // (the HAL builder maps Channel0 X to the X-end field, see program_unpack_x_end).
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .X<FACE_C_DIM - 1>()
        .channel<hal::AddressChannel::Channel1>()
        .X<0>()
        .apply();
}
} // namespace llk_pack_internal_bh

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                  = 4,
    const bool partial_face                        = false)
{
    using namespace hal::sync;
    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
    hal_llk::program_packer_strides(pack_src_format);
    hal_llk::program_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);
}

/**
 * @brief One-time hardware configuration of the packer.
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
    using namespace hal::sync;

    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
    hal_llk::program_packer_strides(pack_src_format);
    hal_llk::program_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);
    hal_llk::program_packer_counters_and_masks();
    hal_llk::program_pack_relu(relu_config);
}

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
 * @brief Pack one tile from Dest to L1.
 *
 * The L1 address is loaded into a GPR and written to the packer configuration with WRCFG, ordered with the
 * pack instruction stream; the packer must have finished the previous tile before its address changes.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "Only PackMode::Default is implemented");
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    // Address-counter instructions issued after a MOP can overtake its expansion (observed on hardware), so wait for
    // the previous tile's expansion before touching the counters. hal::mop::write performs that wait; it rewrites
    // the (unchanged) face count.
    hal::mop::write<hal::mop::Template1Field::OuterLoopCount>(hal_llk::pack_faces_per_tile);

    hal::gpr_ops::set(hal::gpr<hal_llk::PACK_OUTPUT_ADDR_GPR>(), address);
    // The packer must have finished the previous tile before its L1 address changes.
    wait::stall<StallTarget::All, StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<hal_llk::PACK_OUTPUT_ADDR_GPR>());
    hal_llk::start_packer_tile(tile_index); // separates the WRCFG from the first PACR
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
