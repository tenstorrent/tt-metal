// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, pack thread, Blackhole: Dest tiles to L1. Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/).
// A port of _llk_pack_hw_configure_ / configure_pack, _llk_pack_init_ and _llk_pack_ (llk_pack.h, cpack_common.h)
// for 32x32 tiles in the default (tilized) layout.
//
// The calls, in the order they run:
//   pack_params_from(Plan)                  Plan -> pack_params (compile time)
//   pack_configure<P>(l1_acc)               once the packer is idle, programs it completely: input and output
//                                           formats, Dest read mode, ReLU, L1 accumulation (an immediate, or the
//                                           run-time l1_acc with Toggle::Runtime), exponent threshold,
//                                           strides, edge mask, address modifiers, address counters and the MOP
//   pack_execute<P>(dest_row, l1_address)   points the packer at Dest from dest_row, then packs every tile: tile i
//                                           of Dest to tile i in L1
//   pack<On, Plan>(dest_row, l1_address, l1_acc)
//                                           the per-thread call the Compute API makes: configure, then execute,
//                                           only when this build is for thread On (the pack thread)
//
// How one tile is packed. The packer reads Dest through four read interfaces, one Dest row (16 datums) each, so one
// PACR moves 4 rows. Where it reads (tt-isa-documentation, WormholeB0/.../Packers/InputAddressGenerator.md) is
//   Dest row = Offset (DEST_TARGET_REG_CFG_PACK) + (Y * Ystride + Z * Zstride + W * Wstride) / bytes per datum / 16
// with the X, Y, Z, W address counters of channel 0 and the strides in bytes of the input format. Y counts rows in
// a face, Z faces, W tiles. The MOP runs one tile:
//   for each of the 4 faces:          (outer loop)
//     4 PACRs of 4 rows               (inner loop); pack_next_rows moves Y on by 4
//     the last one ends the face:     pack_next_face clears Y and moves Z on (pack_tile_end, on the last face,
//                                     clears both and closes the tile: Last = 1 flushes the output to L1)
// Before each tile, execute sets W to the tile's index and the L1 output address (L1_Dest_addr) to the tile's place.
// The packer writes the output linearly from there (OutputAddressGenerator.md), so the L1 tensor must be dense.
//
// configure assumes nothing about what ran before. Every register is written through the Tensix config unit
// (RMWCIB, SETC16) after a STALLWAIT for the packer to go idle, so a pack still in flight keeps its config, and the
// byte-masked RMWCIB never touches another thread's bits in a shared word (ALU_FORMAT_SPEC, STACC_RELU, THCON REG1).
//
// Not HAL yet: the format helpers masked_data_format (through reg_fmt, llk_lib/formats.h), datum_size_in_bytes,
// IS_A_FORMAT and IS_BFP_A_FORMAT (ckernel_defs.h), L1_ADDRESS and is_valid_L1_address (llk_memory_checks.h) with
// LLK_ASSERT, and the GPR indices p_gpr::ZERO and p_gpr_pack::TMP0 (ckernel_gpr_map.h).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/formats.h"
#include "llk_lib/thread.h"
#include "llk_memory_checks.h"
#include "resolver/plan.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#pragma GCC diagnostic pop

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

// ---- constants -----------------------------------------------------------------------------------------------------

inline constexpr std::uint32_t rows_per_pacr      = 4;        ///< one PACR with all four Dest read interfaces reads 4 Dest rows
inline constexpr std::uint8_t all_read_interfaces = 0;        ///< PACR interface mask 0: all four read interfaces
inline constexpr std::uint32_t zero_relu          = 1;        ///< STACC_RELU ApplyRelu mode 1: negatives to 0 (Packers/ReLU.md)
inline constexpr std::uint32_t no_relu            = 0;        ///< STACC_RELU ApplyRelu mode 0: values pass unchanged
inline constexpr std::uint32_t all_datums_mask    = 0xffff;   ///< PCK_EDGE_OFFSET mask with every datum of a face row kept
inline constexpr std::uint32_t no_downsampling    = 0;        ///< Downsample_mask 0 keeps every datum (Packers/Downsampling.md)
inline constexpr std::uint32_t one_read_per_plane = 1;        ///< pack_reads_per_xy_plane outside reduce (_llk_pack_reduce_mask_clear_)
inline constexpr std::uint32_t packer0_addr_flag  = 1u << 31; ///< L1_Dest_addr bit 31 on packer 0: the other packers' addresses are offsets
                                                              ///< from it (OutputAddressGenerator.md); set as program_packer_destination does
inline constexpr std::uint32_t fp32_exp_bias = 127;           ///< exponent bias of the 8-bit-exponent (B) formats
inline constexpr std::uint32_t bfp_a_min_exp = 14;            ///< minus the smallest normal exponent of the 5-bit-exponent (A) formats: 1 - bias 15
/// Exponent threshold for FP32 Dest to a BFP-A (5-bit exponent) L1 format: datums whose 8-bit exponent is below
/// fp32_exp_bias - bfp_a_min_exp (113) are not representable in A and become 0 (reconfigure_exp_threshold,
/// budabackend#1394).
inline constexpr std::uint32_t bfp_a_exp_threshold = fp32_exp_bias - bfp_a_min_exp;

// The packer address modifier slots the MOP uses (ADDR_MOD_0..2 of the original LLK).
inline constexpr std::uint8_t pack_next_rows = 0; ///< Y on by rows_per_pacr
inline constexpr std::uint8_t pack_tile_end  = 1; ///< Y and Z back to 0: the tile is done
inline constexpr std::uint8_t pack_next_face = 2; ///< Y back to 0, Z on by one face

// ---- parameters ----------------------------------------------------------------------------------------------------

/// Parameters of one pack: Dest to L1, every tile. configure and execute take the same struct.
struct pack_params
{
    DataFormat in_format;     ///< the packer's input format (In_data_format), from the Resolver
    DataFormat out_format;    ///< the format in L1
    bool dest_fp32;           ///< Dest holds 32-bit datums: the packer reads 32 bits per datum
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tiles;      ///< tiles to pack
    std::uint32_t tile_words; ///< one tile in L1, in L1 words: the L1 address step from tile to tile
    bool relu;                ///< negatives to 0 on the way out
    Toggle l1_acc;            ///< add into L1 instead of overwriting it; Runtime: from configure's argument
    bool untilize;            ///< write rows instead of tiles (not ported)
};

/// The pack part of a Plan as pack parameters.
constexpr pack_params pack_params_from(const Plan& p)
{
    return {
        p.pack.in_format,
        p.pack.writes,
        p.pack.dest_fp32,
        p.pack.face_r_dim,
        p.pack.num_faces,
        p.pack.tiles,
        p.pack.tile_bytes / l1_word_bytes,
        p.pack.relu,
        p.pack.l1_acc,
        p.pack.untilize,
    };
}

/// The input format the packer is told: FP8 (E4M3) output goes through the gasket's FP16, so the packer takes FP16
/// (set_packer_config).
constexpr std::uint32_t packer_in_data_format(const pack_params& p)
{
    return p.out_format == DataFormat::Fp8_e4m3 ? reg_fmt(DataFormat::Float16) : reg_fmt(p.in_format);
}

/// Bytes per datum of the packer's input format; the Dest strides count bytes of this size (InputAddressGenerator.md).
constexpr std::uint32_t packer_datum_bytes(const pack_params& p)
{
    return ckernel::datum_size_in_bytes(reg_fmt(p.in_format));
}

// ---- configure -----------------------------------------------------------------------------------------------------

/// How one packer address counter (Y or Z of the Dest side) moves after a PACR.
struct pack_counter_step
{
    std::uint32_t incr; ///< added to the counter
    std::uint32_t clr;  ///< 1: the counter to 0
};

/// Programs packer address modifier Slot: Y (rows in a face) and Z (faces) of the Dest side. The L1 side never moves:
/// the output address comes from L1_Dest_addr and the packer's write pointer.
template <cfg::Sec Slot, pack_counter_step Y, pack_counter_step Z>
inline __attribute__((always_inline)) void set_pack_addr_mod()
{
    using cfg::AddrMod;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::Src][cfg::Y].Incr, Slot, Y.incr>(),
        cfg::set<AddrMod[cfg::Src][cfg::Y].CR, Slot, 0>(), // no carriage return
        cfg::set<AddrMod[cfg::Src][cfg::Y].Clear, Slot, Y.clr>(),
        // The L1 (Dest-named) side's counters never move.
        cfg::set<AddrMod[cfg::Dest][cfg::Y].Incr, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Y].CR, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Y].Clear, Slot, 0>(),
        cfg::set<AddrMod[cfg::Src][cfg::Z].Incr, Slot, Z.incr>(),
        cfg::set<AddrMod[cfg::Src][cfg::Z].Clear, Slot, Z.clr>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Z].Incr, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Z].Clear, Slot, 0>());
}

/// One PACR of the tile, as an instruction word: all four read interfaces, then the counters move by address modifier
/// slot addr_mod; boundary Last closes the tile.
constexpr std::uint32_t pacr(std::uint8_t addr_mod, hal::pack::TileBoundary boundary)
{
    return hal::pack::DataTransfer {.address_modifier = addr_mod, .interfaces = all_read_interfaces, .boundary = boundary}.get_operation();
}

/// The Pack_L1_Acc assignment for P: an immediate for Toggle::Off and Toggle::On (l1_acc unused), the run-time value
/// l1_acc for Toggle::Runtime.
template <pack_params P>
inline __attribute__((always_inline)) auto pack_l1_acc_assignment([[maybe_unused]] bool l1_acc)
{
    using r1 = cfg::ThconReg1Fields;
    if constexpr (P.l1_acc == Toggle::Runtime)
    {
        return cfg::set<r1::Pack_L1_Acc, cfg::Sec::S0>(l1_acc ? 1u : 0u);
    }
    else
    {
        return cfg::set<r1::Pack_L1_Acc, cfg::Sec::S0, P.l1_acc == Toggle::On ? 1u : 0u>();
    }
}

/// Programs the packer for P. Waits for earlier packs to finish first. l1_acc is read only when P.l1_acc is
/// Toggle::Runtime.
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure(bool l1_acc = false)
{
    using namespace hal::counters::adc;
    using cfg::Sec;
    using ckernel::p_gpr;
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the MOP and strides assume four 16-row faces");
    static_assert(!P.untilize, "untilize is not ported yet");

    constexpr std::uint32_t in_format  = packer_in_data_format(P);
    constexpr std::uint32_t out_format = reg_fmt(P.out_format);
    constexpr bool fp8_e4m3            = P.out_format == DataFormat::Fp8_e4m3;
    constexpr bool in_32bit            = is_32bit_format(P.in_format);

    // Nothing below may change while an earlier pack still runs: hold config writes and address counter updates
    // until the packer is idle.
    hal::sync::wait::stall<StallTarget::Config | StallTarget::Tdma, StallCondition::PackerIdle>();

    // Dest-side strides, in bytes of the input format: a face row, a face, a tile. X steps through the counter itself
    // (Xstride 0). The L1 side (register set 1) has no strides.
    constexpr std::uint32_t y_stride = face_cols * packer_datum_bytes(P);
    constexpr std::uint32_t z_stride = P.face_r_dim * y_stride;
    constexpr std::uint32_t w_stride = P.num_faces * z_stride;
    using cfg::PackerReg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg0].Xstride, Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg0].Ystride, Sec::S0, y_stride>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg0].Zstride, Sec::S0, z_stride>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg0].Wstride, Sec::S0, w_stride>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg1].Xstride, Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg1].Ystride, Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg1].Zstride, Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg1].Wstride, Sec::S0, 0>());

    // The Dest format the gasket and ReLU see (Dstacc, no override), and ReLU itself.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::Dstacc_val, Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::Dstacc_override, Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg2::Dstacc, Sec::S0, fp8_e4m3 ? reg_fmt(DataFormat::Float16) : in_format>(),
        cfg::set<cfg::StaccRelu::ApplyRelu, Sec::S0, P.relu ? zero_relu : no_relu>(),
        cfg::set<cfg::StaccRelu::ReluThreshold, Sec::S0, 0>()); // unused: only the threshold modes (2, 3) read it (ReLU.md)

    // How the packer reads Dest: 32-bit datums with a 32-bit Dest or a 32-bit input format; 8-bit integers;
    // unsigned for UINT8 output; mantissa rounded to 10 bits where the conversion needs it (set_packer_config).
    constexpr bool read_32b  = in_32bit || P.dest_fp32;
    constexpr bool read_int8 = !read_32b && (P.in_format == DataFormat::Int8 || P.in_format == DataFormat::UInt8);
    constexpr bool round_10b = (P.dest_fp32 && P.in_format == DataFormat::Float16) || (!ckernel::IS_A_FORMAT(reg_fmt(P.in_format)) && fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, Sec::S0, read_32b ? 1u : 0u>(),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, Sec::S0, P.out_format == DataFormat::UInt8 ? 1u : 0u>(),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, Sec::S0, read_int8 ? 1u : 0u>(),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, Sec::S0, round_10b ? 1u : 0u>());

    // Packer 0's config (THCON SEC0 REG1), every field but L1_Dest_addr (set per tile) and Unp_LF8_4b_exp (the
    // unpacker's). The exponent section: one exponent block per face for block formats; none for FP8 and INT8.
    using r1                     = cfg::ThconReg1Fields;
    constexpr bool no_exponents  = out_format == reg_fmt(DataFormat::Lf8) || out_format == reg_fmt(DataFormat::Int8);
    constexpr bool exp_threshold = P.dest_fp32 && ckernel::IS_BFP_A_FORMAT(out_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r1::Row_start_section_size, Sec::S0, 0>(), // no row-start section: no compression
        cfg::set<r1::Exp_section_size, Sec::S0, no_exponents ? 0u : P.num_faces>());
    // The fields below without a comment are 0, as configure_pack (cpack_common.h) leaves them: it zeroes the whole
    // config before setting the formats and section sizes.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r1::Disable_zero_compress, Sec::S0, 1>(),   // plain tiles, no zero compression
        cfg::set<r1::Add_l1_dest_addr_offset, Sec::S0, 0>(), // the address is L1_Dest_addr alone
        cfg::set<r1::Disable_pack_zero_flags, Sec::S0, 0>(),
        cfg::set<r1::ovrd_default_throttle_mode, Sec::S0, 0>(),
        cfg::set<r1::Out_data_format, Sec::S0, out_format>(),
        cfg::set<r1::In_data_format, Sec::S0, in_format>(),
        cfg::set<r1::Dis_shared_exp_assembler, Sec::S0, 0>(),
        cfg::set<r1::Auto_set_last_pacr_intf_sel, Sec::S0, 0>(),
        cfg::set<r1::Enable_out_fifo, Sec::S0, 0>(),
        cfg::set<r1::Sub_l1_tile_header_size, Sec::S0, 0>(),    // 0: nothing to subtract; L1_ADDRESS already points one word before the data
        cfg::set<r1::Source_interface_selection, Sec::S0, 0>(), // read Dest, not L1
        cfg::set<r1::pack_start_intf_pos, Sec::S0, 0>(),
        cfg::set<r1::All_pack_disable_zero_compress_ovrd, Sec::S0, 0>(),
        cfg::set<r1::Add_tile_header_size, Sec::S0, 0>(),
        cfg::set<r1::pack_dis_y_pos_start_offset, Sec::S0, 0>(),
        cfg::set<r1::L1_source_addr, Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r1::Downsample_mask, Sec::S0, no_downsampling>(),
        cfg::set<r1::Downsample_rate, Sec::S0, 0>(), // as configure_pack: config zeroed
        pack_l1_acc_assignment<P>(l1_acc),
        cfg::set<r1::Exp_threshold_en, Sec::S0, exp_threshold ? 1u : 0u>(),
        cfg::set<r1::Pac_LF8_4b_exp, Sec::S0, fp8_e4m3 ? 1u : 0u>(),
        cfg::set<r1::Exp_threshold, Sec::S0, exp_threshold ? bfp_a_exp_threshold : 0u>());

    // Edge masking off: one read per XY plane, every datum kept, every row on mask 0 (_llk_pack_reduce_mask_clear_).
    // The other PACK_COUNTERS fields are 0, as configure_pack writes them (the whole word 0, then
    // pack_reads_per_xy_plane = 1): no automatic last-PACR generation.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, Sec::S0, one_read_per_plane>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, Sec::S0, all_datums_mask>(),
        cfg::set<cfg::PckEdgeMode::mode, Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, Sec::S0, 0>());
    // Every 2-bit entry of TILE_ROW_SET_MAPPING_0 to 0: the whole word from the always-zero GPR.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[0][0], Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<p_gpr::ZERO>());

    // Address modifiers for the MOP below.
    constexpr pack_counter_step stay {0, 0};
    constexpr pack_counter_step clear {0, 1};
    constexpr pack_counter_step next_face {1, 0}; // Z counts faces
    set_pack_addr_mod<static_cast<Sec>(pack_next_rows), pack_counter_step {rows_per_pacr, 0}, stay>();
    set_pack_addr_mod<static_cast<Sec>(pack_tile_end), clear, clear>();
    set_pack_addr_mod<static_cast<Sec>(pack_next_face), clear, next_face>();

    // Address counters to 0, except channel 1's X: the last datum of a row, so each read interface takes one
    // 16-datum row (InputNumDatums = channel 1 X - channel 0 X + 1).
    hal::counters::set<
        AddrCounter {.channel0 = {.x = 0, .y = 0, .z = 0, .w = 0}, .channel1 = {.x = face_cols - 1, .y = 0, .z = 0, .w = 0}}.for_clients<Client::Packers>()>();

    // MOP template 1: one tile. Outer loop: the faces; inner loop: the PACRs of a face.
    constexpr std::uint32_t next_rows_op = pacr(pack_next_rows, hal::pack::TileBoundary::NotLast);
    constexpr std::uint32_t next_face_op = pacr(pack_next_face, hal::pack::TileBoundary::NotLast);
    constexpr std::uint32_t tile_end_op  = pacr(pack_tile_end, hal::pack::TileBoundary::Last);
    hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
        .outer_loop = {.count = P.num_faces},
        .inner_loop = {
            .count                               = P.face_r_dim / rows_per_pacr,
            .body_op                             = next_rows_op,
            .last_op_on_final_outer_iteration    = tile_end_op,
            .last_op_on_nonfinal_outer_iteration = next_face_op}}>();
}

// ---- execute -------------------------------------------------------------------------------------------------------

/// Packs every tile of the Dest tensor starting at Dest row dest_row (absolute) to the L1 tensor at byte address
/// l1_address: tile i of Dest (row-major) to tile i of L1.
template <pack_params P>
inline __attribute__((always_inline)) void pack_execute(std::uint32_t dest_row, std::uint32_t l1_address)
{
    using namespace hal::counters::adc;

    // The packer reads Dest from dest_row on. Earlier packs read the old offset: wait for them first.
    // The register is written whole from a GPR (Offset in the low bits, ZOffset 0): a byte-wise RMWCIB would read it
    // back first, which ttsim does not support for this register.
    static_assert(cfg::DestTargetRegCfgPack::Offset.shamt(cfg::Sec::S0) == 0, "Offset is the low field of the word");
    LLK_ASSERT(dest_row < (1u << cfg::DestTargetRegCfgPack::Offset.width), "Dest row does not fit the packer's Dest offset");
    hal::scalar::set(hal::gpr<ckernel::p_gpr_pack::TMP0>(), dest_row);
    hal::sync::wait::stall<StallTarget::Config, StallCondition::PackerIdle | StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<ckernel::p_gpr_pack::TMP0>());

    const std::uint32_t first_tile = L1_ADDRESS(l1_address); // 16-byte words, one before the data (the tile header slot)
    for (std::uint32_t tile = 0; tile < P.tiles; tile++)
    {
        const std::uint32_t tile_address = first_tile + tile * P.tile_words;
        LLK_ASSERT(is_valid_L1_address(tile_address), "pack address is outside L1");
        // W selects the Dest tile; the tile's first PACR samples the L1 address (in-order behind the last tile's
        // PACRs, as _llk_pack_ relies on). The STALLWAIT keeps the PACRs from starting before the address lands.
        hal::counters::set(AddrCounter {.channel0 = {.w = tile}}.for_clients<Client::Packers>());
        cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(packer0_addr_flag | tile_address);
        hal::sync::wait::stall<StallTarget::Pack, StallCondition::ConfigUnitIdle>();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        // Y and Z are back at 0 (tile_end); the L1 side's counters never moved.
    }
}

/// The pack thread's part of a Compute API pack. dest_row is the Dest tensor's first row (absolute), l1_address the
/// byte address of the L1 tensor, l1_acc the run-time L1 accumulation (read only when the Plan's l1_acc is
/// Toggle::Runtime).
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void pack(std::uint32_t dest_row, std::uint32_t l1_address, bool l1_acc = false)
{
    static_assert(On == Thread::Pack, "the packer is driven from the pack thread");
    constexpr pack_params p = pack_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        pack_configure<p>(l1_acc);
        pack_execute<p>(dest_row, l1_address);
    }
}

} // namespace llk2::llk_lib
