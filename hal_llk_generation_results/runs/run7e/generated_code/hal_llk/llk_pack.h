// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole pack LLK (Dest -> L1, default tile layout).

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
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

namespace hal_llk::pack
{

// Packer ADDR_MOD slots.
constexpr std::uint8_t AM_NEXT_ROW  = 0; // Next Dest row of the same face.
constexpr std::uint8_t AM_NEXT_FACE = 1; // Last row of a face: rewind the row, step to the next face.
constexpr std::uint8_t AM_TILE_DONE = 2; // Last row of the tile: rewind row and face.

constexpr hal::pack::DataTransfer row_transfer(const std::uint8_t address_modifier, const bool zero_output, const hal::pack::TileBoundary boundary)
{
    return {
        .address_modifier = address_modifier,
        .interfaces       = 0b0001, // One Dest-read interface: one row of FACE_C_DIM datums.
        .datum_override   = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None,
        .boundary         = boundary,
    };
}

constexpr bool is_32bit_dest_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0xF;
    return f == static_cast<std::uint32_t>(DataFormat::Float32) || f == static_cast<std::uint32_t>(DataFormat::Int32) ||
           f == static_cast<std::uint32_t>(DataFormat::UInt32) || f == static_cast<std::uint32_t>(DataFormat::Tf32);
}

constexpr bool is_bfp_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0xF;
    return f == static_cast<std::uint32_t>(DataFormat::Bfp8) || f == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           f == static_cast<std::uint32_t>(DataFormat::Bfp4) || f == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           f == static_cast<std::uint32_t>(DataFormat::Bfp2) || f == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
}

constexpr bool is_bfp_a_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0xF;
    return f == static_cast<std::uint32_t>(DataFormat::Bfp8) || f == static_cast<std::uint32_t>(DataFormat::Bfp4) ||
           f == static_cast<std::uint32_t>(DataFormat::Bfp2);
}

// Dest addressing strides of the packer read side, in bytes of the packer source format.
inline void set_packer_strides(const std::uint32_t pack_src_format)
{
    namespace cfg                 = hal::cfg;
    const std::uint32_t x_stride  = ((pack_src_format & 0x3) == static_cast<std::uint32_t>(DataFormat::Float32))   ? 4
                                    : ((pack_src_format & 0x3) == static_cast<std::uint32_t>(DataFormat::Float16)) ? 2
                                                                                                                  : 1;
    const std::uint32_t y_stride  = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride  = FACE_R_DIM * y_stride;
    const std::uint32_t w_stride  = TILE_NUM_FACES * z_stride;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Ystride, cfg::Sec::S0>(y_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Zstride, cfg::Sec::S0>(z_stride),
        cfg::set<cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0].Wstride, cfg::Sec::S0>(w_stride));
}

} // namespace hal_llk::pack

template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "Only the default pack layout is implemented");
    constexpr hal::pack::AddressModifier next_row {
        .selection  = hal::cfg::Sec::S0,
        .source_row = {.step = 1},
    };
    constexpr hal::pack::AddressModifier next_face {
        .selection   = hal::cfg::Sec::S1,
        .source_row  = {.clear = true},
        .source_face = {.step = 1},
    };
    constexpr hal::pack::AddressModifier tile_done {
        .selection   = hal::cfg::Sec::S2,
        .source_row  = {.clear = true},
        .source_face = {.clear = true},
    };
    hal::pack::configure_address_modifiers<next_row, next_face, tile_done>();
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const std::uint32_t num_tiles = 1)
{
    using namespace hal_llk::pack;
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;
    static_assert(pack_mode == PackMode::Default, "Only the default pack layout is implemented");
    LLK_ASSERT(num_tiles == 1, "Only one tile per pack MOP run is implemented");

    // One PACR per Dest row; the last row of each face steps to the next face, the last row of the tile closes it.
    MopConfig<MopTemplate::Template1> config {};
    config.outer_loop.count = num_faces;
    config.inner_loop.count = face_r_dim;
    config.inner_loop.body_op = row_transfer(AM_NEXT_ROW, zero_output, hal::pack::TileBoundary::NotLast).get_operation();
    config.inner_loop.last_op_on_nonfinal_outer_iteration = row_transfer(AM_NEXT_FACE, zero_output, hal::pack::TileBoundary::NotLast).get_operation();
    config.inner_loop.last_op_on_final_outer_iteration    = row_transfer(AM_TILE_DONE, zero_output, hal::pack::TileBoundary::Last).get_operation();
    hal::mop::program(config);
}

namespace llk_pack_internal_bh
{
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    const std::uint32_t pack_src_format, const std::uint32_t face_r_dim, const std::uint32_t tile_c_dim, const std::uint32_t num_faces, const std::uint32_t num_tiles)
{
    static_assert(!mutex_ADC, "mutex_ADC is not supported by this implementation");
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
        hal_llk::pack::set_packer_strides(pack_src_format);
    }
    // One row (FACE_C_DIM datums) per PACR.
    hal_llk::pack::set_packer_x_range();
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
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    using namespace hal_llk::pack;
    static_assert(pack_mode == PackMode::Default, "Only the default pack layout is implemented");

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    set_packer_strides(pack_src_format);

    // Packer formats and output section layout (packer register set 0).
    const std::uint32_t dst_fmt          = pack_dst_format & 0xF;
    // Only block-float outputs carry a shared-exponent section ahead of the data.
    const std::uint32_t exp_section_size = is_bfp_format(pack_dst_format) ? (partial_face ? 1 : num_faces) : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Row_start_section_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_section_size, cfg::Sec::S0>(exp_section_size),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_zero_compress, cfg::Sec::S0, 1>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Out_data_format, cfg::Sec::S0>(dst_fmt),
        cfg::set<cfg::Thcon[cfg::Reg1].In_data_format, cfg::Sec::S0>(pack_src_format & 0xF),
        cfg::set<cfg::Thcon[cfg::Reg1].Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Sub_l1_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Source_interface_selection, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>());

    // FP8 (E4M3) output, L1 accumulation off, exponent thresholding for FP32 Dest -> BFP(A).
    const bool lf8_4b_exp     = (pack_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);
    const bool exp_threshold  = is_fp32_dest_acc_en && is_bfp_a_format(pack_dst_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg1].Pac_LF8_4b_exp, cfg::Sec::S0>(lf8_4b_exp ? 1 : 0),
        cfg::set<cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold_en, cfg::Sec::S0>(exp_threshold ? 1 : 0),
        cfg::set<cfg::Thcon[cfg::Reg1].Exp_threshold, cfg::Sec::S0>(exp_threshold ? 113 : 0));

    // How Dest is read: 32-bit datums when Dest accumulates in 32 bits or holds a 32-bit format.
    const bool read_32b  = is_fp32_dest_acc_en || is_32bit_dest_format(pack_src_format);
    // Read_int8 selects the SFPU int8 Dest layout; FPU int8 results are read as plain 32-bit Dest data.
    const bool read_int8 = false;
    const bool read_uint = (pack_src_format & 0xF) == static_cast<std::uint32_t>(DataFormat::UInt8);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(read_32b ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(read_uint ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>(read_int8 ? 1 : 0),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, 0>());

    // One Dest read per XY plane.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());

    // No edge masking: every row maps to edge-mask set 0, which passes all datums.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 0xFFFF>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1 /* word count */>(std::array<std::uint32_t, 1> {0});

    // ReLU.
    _llk_pack_relu_config_(ckernel::ReluConfig::from_packed(relu_config));
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

template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    static_assert(pack_mode == PackMode::Default, "Only the default pack layout is implemented");

    set_dst_write_addr<mutex_ADC>(tile_index);

    // The previous tile must be fully packed before its L1 destination is replaced.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(address);
    // The packer must not start before the new destination has landed.
    sync::wait::stall<sync::StallTarget::Pack, sync::StallCondition::ConfigUnitIdle>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
