// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, pack thread, pack on Blackhole: Dest tiles to L1, 32x32 tiles in the default (tilized) layout. Written only
// in HAL calls (tt_llk_blackhole/llk_lib/hal/), except where noted.
//
//   pack_params                    one struct; pack_configure and pack_execute take the same one
//   pack_params_from(Plan)         fills it from the Resolver's Plan
//   pack_configure<P>()            once the packer is idle, programs it completely: the Dest row stride, input and
//                                  output formats, Dest read mode, ReLU, L1 accumulation (off for Toggle::Runtime),
//                                  exponent threshold, edge mask, the two address modifiers, the MOP of one run
//                                  (per-tile form: one tile; block form: every tile; sum form: every tile but the
//                                  first) and the address counters
//   pack_execute<P>(dest_row, l1_address, tiles, l1_acc)
//                                  packs every tile of the Dest tensor at dest_row: tile i of Dest to tile i in L1;
//                                  with sum_tiles, every tile to the one tile at l1_address, the second one on with
//                                  L1 accumulation. tiles is their count: a ct (P.tiles) or, for a size known
//                                  only at run time, an rt of at most that (resolver/work_counts.h): the block form
//                                  and the sum form then program the MOP here, with the count, instead of in
//                                  configure. l1_acc is the run-time value of a Toggle::Runtime l1_acc
//   pack<On, Plan>(counts, dest_row, l1_address, l1_acc)
//                                  the per-thread call the Compute API makes: configures and executes, only when this
//                                  build is for thread On (the pack thread). counts (WorkCounts) is the run-time side of
//                                  the Plan's Work
//
// The caller takes the source and the L1 tensor over for the packer before the entry (llk_lib/sync.h), as for every
// op, so nothing here is issued while the unit that wrote the source still works on it.
//
// Departures from the shape (llk_lib/thread.h), and why:
//   - execute writes one config bit, Pack_L1_Acc, in two cases, both while the packer is idle. With Toggle::Runtime
//     the kernel chooses L1 accumulation per call, so the value is a run-time argument, which only execute takes:
//     configure writes the bit off and execute turns it on when asked. With sum_tiles the bit changes between the
//     first tile and the rest, inside the call's work.
//   - with a run-time tile count, the block and the sum form program the MOP in execute, not in configure: the MOP's
//     outer loop count is the tile count, which only execute is given. configure<P> and execute<P> decide this from
//     the same P (pack_runtime_mop); execute checks that its count's type agrees.
//
// How one tile is packed. The packer reads Dest through four read interfaces, one Dest row (16 datums) each, so one
// PACR moves 4 rows. Where it reads (tt-isa-documentation, WormholeB0/.../Packers/InputAddressGenerator.md) is
//   Dest row = Offset (DEST_TARGET_REG_CFG_PACK) + (Y * Ystride + Z * Zstride + W * Wstride) / bytes per datum / 16
// with the X, Y, Z, W address counters of channel 0 and the strides in bytes of the input format. Only Y is used: it
// counts Dest rows from the tensor's first row, through the faces of a tile and on through the tiles (a tensor's
// tiles follow each other in Dest), and Ystride is one row. Z and W stay 0, so their strides are not written. A tile
// is 16 PACRs of 4 rows; pack_next_rows moves Y on by 4, and the PACR that ends a run clears Y (pack_run_end) and
// closes the output (Last = 1 flushes it to L1). The MOP's inner loop is one tile, its outer loop the tiles of a run.
// In the per-tile form execute sets, before each tile, Y to the tile's first row and the L1 output address
// (L1_Dest_addr) to the tile's place. The packer writes the output linearly from there (OutputAddressGenerator.md),
// so the L1 tensor must be dense.
//
// The three forms. execute packs in one of three ways, chosen at compile time (pack_execute_per_tile, _block, _sum):
//
//               L1 address written   Last on                  Y set by execute   MOP's outer loop
//   per-tile    per tile             every tile's last PACR   per tile           1 tile
//   block       once                 the run's final PACR     no: it runs on     every tile
//   sum_tiles   once (one L1 tile)   every tile's last PACR   no: it runs on     every tile but the first
//
//   execute issues, after the Dest row:
//   per-tile    for each tile: Y, the address, a wait for it, one MOP run
//   block       the address, one MOP run
//   sum_tiles   the address, the first tile's PACRs, the Pack_L1_Acc switch, one MOP run
//
// The per-tile form is the one described above; the other two:
//
// Block form (pack_in_blocks). The packer takes its output address from L1_Dest_addr only at the first PACR after a
// Last and writes linearly from there, and a dense L1 tensor in a format without a per-tile exponent section is its
// tiles' datums one after another. So configure programs the MOP's outer loop over every tile (Y runs on from tile to
// tile, the only Last is on the final PACR) and execute writes the Dest offset and the first tile's address once and
// issues one MOP run. Block float L1 formats keep the per-tile form (pack_in_blocks has the reason).
//
// sum_tiles (every Dest tile into one L1 tile): the L1 address does not step. It is written once, for the first tile:
// a tile's closing PACR (Last) makes the packer take its output address from L1_Dest_addr again at the next PACR
// (OutputAddressGenerator.md, NeedsNewAddress), and the L1 side's counters never move, so every tile starts at the
// same place. The first tile goes out as configure set Pack_L1_Acc; before the second, execute waits for the packer
// and turns Pack_L1_Acc on (the addition is done at L1's port, datum by datum: Packers/Accumulation.md), so the L1
// tile ends as tile 0 + ... + tile n-1 in the L1 format (plus what it held, when the first tile accumulated too).
// On the Dest side nothing is set per tile: Y runs on through the rows of all the tiles, and every tile's last PACR
// has Last. The first tile's PACRs are issued one by one (as fast as a MOP expands them) and the MOP holds one run
// over all the other tiles, so the switch of Pack_L1_Acc sits between the two and the MOP is programmed once.
//
// configure assumes nothing about what ran before. Every register is written through the Tensix config unit
// (RMWCIB, SETC16) after a STALLWAIT for the packer to go idle, so a pack still in flight keeps its config, and the
// byte-masked RMWCIB never touches another thread's bits in a shared word (ALU_FORMAT_SPEC, STACC_RELU, THCON REG1).
//
// Shared config words and waits (the tests named keep it checked):
//   - pack_configure writes, besides the packer's own registers, ALU_FORMAT_SPEC_REG Dstacc_val / Dstacc_override,
//     ALU_FORMAT_SPEC_REG2 Dstacc and STACC_RELU ApplyRelu after a wait for the packer only, while the math thread's
//     next op may run on the FPU or SFPU. That is enough: only the packer reads them. In tt-isa-documentation the
//     three fields occur only on the packer's pages (Packers/ReLU.md, Packers/FormatConversion.md, PACR.md); no
//     Matrix Unit or Vector Unit instruction's model names them, and on Blackhole REG2 Dstacc has no effect at all
//     (ReLU.md: the intermediate format is the packer's In_data_format; it is written as 1.0's configure_pack
//     writes it). The bytes they share with FPU / SFPU fields are written by RMWCIB with pack's bits as the mask,
//     atomic per byte (RMWCIB.md). tests: test_llk2_pack_shared_config.
//   - Pack_L1_Acc is never written while the packer works. The ISA does not say when a PACR samples the bit
//     (PACR.md: some fields "at an unspecified point during the lifetime of a PACR"), and 1.0 waits
//     (reconfigure_packer_l1_acc). The device takes it per PACR at issue (measured PACR by PACR,
//     test_llk2_pack_l1_acc_latch), but that is a measurement on one board, so the waits must stay, also through a
//     compiler that tracks Tensix state and deletes waits it finds to hold nothing. Two things keep them there
//     (do not change one without looking at a pack thread's ELF built with such a compiler):
//       1. One wait for the packer per pack: pack_configure's. pack_execute's wait in front of the Dest offset
//          waits for the scalar unit only. With a second packer-idle wait behind it, such a compiler deletes
//          configure's whenever the only write left of a configure is the Pack_L1_Acc word, and the word is then
//          written one instruction behind the previous pack's MOP.
//       2. The sum form's switch waits with the packer held too (Config | Pack): a wait that holds only config
//          writes is deleted when PACRs are the last thing issued.
//
// Ported from: _llk_pack_hw_configure_ / configure_pack, _llk_pack_init_, _llk_pack_mop_config_ and _llk_pack_
// (llk_pack.h, cpack_common.h); sum_tiles from SDPA 1.0's pack_tile<true>(dst_index++, sum_cb, row) under
// llk_pack_reconfig_l1_acc.
//
// Not HAL: LLK_ASSERT (llk_assert.h).

#pragma once

#include <cstdint>
#include <utility>

#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_lib/formats.h"
#include "llk_lib/platform/format.h"
#include "llk_lib/platform/gpr_map.h"
#include "llk_lib/platform/l1.h"
#include "llk_lib/thread.h"
#include "resolver/plan.h"
#include "resolver/work_counts.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#include "llk_lib/platform/limits.h"
#pragma GCC diagnostic pop

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

// ---- parameters -----------------------------------------------------------------------------------------------------

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
    Toggle l1_acc;            ///< add into L1 instead of overwriting it; Runtime: from execute's argument
    bool untilize;            ///< write rows instead of tiles (not ported)
    bool sum_tiles;           ///< every tile to the one L1 tile at the address (no address step); l1_acc is then the
                              ///< first tile's, and the rest accumulate
    bool runtime_tiles = false; ///< tiles is an upper bound: execute takes the count (resolver/work_counts.h)
};

/// The pack part of a Plan as pack parameters.
constexpr pack_params pack_params_from(const Plan& plan)
{
    return {
        .in_format     = plan.pack.in_format,
        .out_format    = plan.pack.writes,
        .dest_fp32     = plan.pack.dest_fp32,
        .face_r_dim    = plan.pack.face_r_dim,
        .num_faces     = plan.pack.num_faces,
        .tiles         = plan.work.tile_rows * plan.work.tile_cols,
        .tile_words    = plan.pack.tile_bytes / l1_word_bytes,
        .relu          = plan.pack.relu,
        .l1_acc        = plan.pack.l1_acc,
        .untilize      = plan.pack.untilize,
        .sum_tiles     = plan.pack.sum_tiles,
        .runtime_tiles = plan.work.rows_from != CountSource::Fixed || plan.work.cols_from != CountSource::Fixed,
    };
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

inline constexpr std::uint32_t pack_rows_per_pacr      = 4;        ///< one PACR with all four Dest read interfaces reads 4 Dest rows
inline constexpr std::uint8_t pack_all_read_interfaces = 0;        ///< PACR interface mask 0: all four read interfaces
inline constexpr std::uint32_t pack_zero_relu          = 1;        ///< STACC_RELU ApplyRelu mode 1: negatives to 0 (Packers/ReLU.md)
inline constexpr std::uint32_t pack_no_relu            = 0;        ///< STACC_RELU ApplyRelu mode 0: values pass unchanged
inline constexpr std::uint32_t pack_all_datums_mask    = 0xffff;   ///< PCK_EDGE_OFFSET mask with every datum of a face row kept
inline constexpr std::uint32_t pack_no_downsampling    = 0;        ///< Downsample_mask 0 keeps every datum (Packers/Downsampling.md)
inline constexpr std::uint32_t pack_one_read_per_plane = 1;        ///< pack_reads_per_xy_plane outside reduce (_llk_pack_reduce_mask_clear_)
inline constexpr std::uint32_t pack_packer0_addr_flag  = 1u << 31; ///< L1_Dest_addr bit 31 on packer 0: the other packers' addresses are offsets
                                                                   ///< from it (OutputAddressGenerator.md); set as program_packer_destination does
inline constexpr std::uint32_t pack_fp32_exp_bias = 127;           ///< exponent bias of the 8-bit-exponent (B) formats
inline constexpr std::uint32_t pack_bfp_a_min_exp = 14;            ///< minus the smallest normal exponent of the 5-bit-exponent (A) formats: 1 - bias 15
/// Exponent threshold for FP32 Dest to a BFP-A (5-bit exponent) L1 format: datums whose 8-bit exponent is below
/// pack_fp32_exp_bias - pack_bfp_a_min_exp (113) are not representable in A and become 0 (reconfigure_exp_threshold).
inline constexpr std::uint32_t pack_bfp_a_exp_threshold = pack_fp32_exp_bias - pack_bfp_a_min_exp;

// The packer address modifier slots the PACRs name.
inline constexpr std::uint8_t pack_next_rows = 0; ///< Y on by pack_rows_per_pacr
inline constexpr std::uint8_t pack_run_end   = 1; ///< Y back to 0: the run is done

/// The input format the packer is told: FP8 (E4M3) output goes through the gasket's FP16, so the packer takes FP16
/// (set_packer_config).
constexpr std::uint32_t pack_in_data_format(const pack_params& params)
{
    return params.out_format == DataFormat::Fp8_e4m3 ? format_id(DataFormat::Float16) : format_id(params.in_format);
}

/// Bytes per datum of the packer's input format; the Dest strides count bytes of this size (InputAddressGenerator.md).
constexpr std::uint32_t pack_datum_bytes(const pack_params& params)
{
    return platform::format::datum_bytes(params.in_format);
}

/// Whether params' tiles go out as one block: one L1 address and one MOP run for all of them. Only for
/// L1 formats whose tile is nothing but its datums, so that consecutive tiles are consecutive data, and of those the
/// ones with 16- or 32-bit datums: Float32, Int32, Float16, Float16_b, UInt16. For them the packer has one output
/// stream (OutputAddressGenerator.md).
/// The rest keep one address and one Last per tile:
///   - block floats (Bfp8, Bfp4, Bfp2, A or B): a tile starts with its exponent section, which the packer writes
///     through a second stream. That stream starts at the address the first PACR samples, the data stream
///     Exp_section_size words after it, and both run linearly: in one block over several tiles the exponents of tile
///     1 would land on the mantissas of tile 0.
///   - Int8, UInt8: a tile is only its datums (Exp_section_size is 0), so by the same model a block would be right,
///     but they go through the address generator's branch for formats under 16 bits, as the block floats do, and no
///     test packs to them yet.
constexpr bool pack_in_blocks(const pack_params& params)
{
    // A sum pack (sum_tiles) sends every tile to the SAME L1 tile: each tile needs its own Last, so the per-tile form.
    return datum_bits(params.out_format) >= 16 && !params.sum_tiles;
}

/// PACRs that pack one tile: pack_rows_per_pacr rows each.
constexpr std::uint32_t pack_pacrs_per_tile(const pack_params& params)
{
    return params.num_faces * params.face_r_dim / pack_rows_per_pacr;
}

/// Tiles one MOP run packs, the MOP's outer loop count: all of them in block form; all but the first in the sum form
/// (execute issues the first tile's PACRs itself; 0: no MOP); 1 in the per-tile form. Y counts the rows of a run's
/// tiles (13 bits: 127 full tiles) and platform::mop_max_loop_count is higher still; Dest holds 16 tiles.
constexpr std::uint32_t pack_run_tiles(const pack_params& params)
{
    return pack_in_blocks(params) ? params.tiles : params.sum_tiles ? params.tiles - 1 : 1;
}

// ---- configure ------------------------------------------------------------------------------------------------------

/// How the packer's Dest-side Y counter (Dest rows) moves after a PACR.
struct pack_counter_step
{
    std::uint32_t incr; ///< added to the counter
    std::uint32_t clr;  ///< 1: the counter to 0
};

/// Programs packer address modifier SlotNumber (the number a PACR names it by): Y of the Dest side moves, nothing else does. Z of the Dest side stays 0,
/// and the L1 side never moves: the output address comes from L1_Dest_addr and the packer's write pointer.
template <std::uint8_t SlotNumber, pack_counter_step Y>
inline __attribute__((always_inline)) void pack_set_addr_mod()
{
    using cfg::AddrMod;
    static_assert(SlotNumber < 8, "there are 8 address modifier slots");
    constexpr cfg::Sec Slot = static_cast<cfg::Sec>(SlotNumber); // the HAL names the slot's config section
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::Src][cfg::Y].Incr, Slot, Y.incr>(),
        cfg::set<AddrMod[cfg::Src][cfg::Y].CR, Slot, 0>(), // no carriage return
        cfg::set<AddrMod[cfg::Src][cfg::Y].Clear, Slot, Y.clr>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Y].Incr, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Y].CR, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Y].Clear, Slot, 0>(),
        cfg::set<AddrMod[cfg::Src][cfg::Z].Incr, Slot, 0>(),
        cfg::set<AddrMod[cfg::Src][cfg::Z].Clear, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Z].Incr, Slot, 0>(),
        cfg::set<AddrMod[cfg::Dest][cfg::Z].Clear, Slot, 0>());
}

/// Every 2-bit entry of TILE_ROW_SET_MAPPING_0 to 0 (every face row on edge mask set 0): one word, immediate writes.
template <std::size_t... Entries>
inline __attribute__((always_inline)) void pack_clear_row_set_mapping(std::index_sequence<Entries...>)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::TileRowSetMapping[0][Entries], cfg::Sec::S0, 0>()...);
}

/// One PACR of a tile: all four read interfaces, then the counters move by address modifier slot addr_mod; boundary
/// Last closes the output (the tile, or in block form the run).
constexpr hal::pack::DataTransfer pack_transfer(std::uint8_t addr_mod, hal::pack::TileBoundary boundary)
{
    return {.address_modifier = addr_mod, .interfaces = pack_all_read_interfaces, .boundary = boundary};
}

/// The same PACR as an instruction word, for the MOP.
constexpr std::uint32_t pack_pacr(std::uint8_t addr_mod, hal::pack::TileBoundary boundary)
{
    return pack_transfer(addr_mod, boundary).get_operation();
}

/// L1_Dest_addr's value for an output that starts at L1 word word_address (16-byte words).
constexpr std::uint32_t pack_l1_dest_addr(std::uint32_t word_address)
{
    return pack_packer0_addr_flag | word_address;
}

/// Packer 0's config word with Pack_L1_Acc (THCON SEC0 REG1, third word), every field a constant: downsampling off,
/// L1 accumulation L1Acc, and the exponent threshold and 4-bit-exponent FP8 fields of P's formats.
template <pack_params P, std::uint32_t L1Acc>
inline __attribute__((always_inline)) void pack_write_l1_acc_word()
{
    using r1                     = cfg::ThconReg1Fields;
    constexpr bool fp8_e4m3      = P.out_format == DataFormat::Fp8_e4m3;
    constexpr bool exp_threshold = P.dest_fp32 && platform::format::is_block_float_5bit_exponent(P.out_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r1::Downsample_mask, cfg::Sec::S0, pack_no_downsampling>(),
        cfg::set<r1::Downsample_rate, cfg::Sec::S0, 0>(), // as configure_pack: config zeroed
        cfg::set<r1::Pack_L1_Acc, cfg::Sec::S0, L1Acc>(),
        cfg::set<r1::Exp_threshold_en, cfg::Sec::S0, exp_threshold ? 1u : 0u>(),
        cfg::set<r1::Pac_LF8_4b_exp, cfg::Sec::S0, fp8_e4m3 ? 1u : 0u>(),
        cfg::set<r1::Exp_threshold, cfg::Sec::S0, exp_threshold ? pack_bfp_a_exp_threshold : 0u>());
}

/// Whether the MOP's outer loop count is a run-time value: a run-time tile count in the block or the sum form (the
/// per-tile form's run is one tile whatever the count). configure then leaves the MOP to execute, which programs it
/// with the count: once per call, as configure does for a compile-time count.
constexpr bool pack_runtime_mop(const pack_params& params)
{
    return params.runtime_tiles && (pack_in_blocks(params) || params.sum_tiles);
}

/// The MOP of one run of run_tiles tiles of P. Inner loop: the PACRs of a tile; outer loop: the tiles of the run.
/// Every PACR moves Y on by its rows (the tiles follow each other in Dest), the run's final one clears Y and closes
/// the output (Last). In the sum form every tile's last PACR has Last: each tile goes to the same L1 tile.
constexpr auto pack_run_mop(const pack_params& params, const std::uint32_t run_tiles)
{
    constexpr std::uint32_t next_rows_op = pack_pacr(pack_next_rows, hal::pack::TileBoundary::NotLast);
    constexpr std::uint32_t tile_end_op  = pack_pacr(pack_next_rows, hal::pack::TileBoundary::Last);
    constexpr std::uint32_t run_end_op   = pack_pacr(pack_run_end, hal::pack::TileBoundary::Last);
    const auto run_mop = hal::mop::loop(run_tiles, hal::mop::loop(pack_pacrs_per_tile(params), next_rows_op)).patch(hal::mop::last_outer, run_end_op);
    // The last PACR of every tile but the run's last is the inner loop's own op (next_rows_op) unless the tiles are
    // summed. A run of one tile issues no such PACR, and the HAL refuses a patch that is never issued.
    return params.sum_tiles && run_tiles != 1 ? run_mop.patch(hal::mop::last_inner, tile_end_op) : run_mop;
}

/// MOP template 1 for a run of run_tiles tiles, a run-time count (pack_runtime_mop): execute programs it, where the
/// count is known.
template <pack_params P>
inline __attribute__((always_inline)) void pack_program_runtime_mop(const std::uint32_t run_tiles)
{
    hal::mop::program(pack_run_mop(P, run_tiles));
}

/// MOP template 1 for P, a compile-time count: one run of pack_run_tiles(P) tiles (a sum of one tile programs no
/// MOP). With a run-time count configure leaves the MOP to execute (pack_program_runtime_mop).
template <pack_params P>
inline __attribute__((always_inline)) void pack_program_mop()
{
    static_assert(pack_run_tiles(P) <= platform::mop_max_loop_count, "a run's tiles must fit the MOP's outer loop count");
    if constexpr (pack_run_tiles(P) != 0 && !pack_runtime_mop(P))
    {
        constexpr auto run_mop = pack_run_mop(P, pack_run_tiles(P));
        hal::mop::program(run_mop);
    }
}

/// Configure, the Dest-side Y stride, in bytes of the input format: one Dest row. X steps through the counter itself
/// (Xstride 0). No other stride is written, because the counter it multiplies stays 0 for the whole op, whatever the
/// register holds: configure sets every counter to 0 (pack_configure_counters), and the address modifiers move only
/// the Dest side's Y (pack_set_addr_mod). That is the Dest side's Z and W, and the L1 side (register set 1), whose
/// strides multiply channel 1's Y, Z and W (OutputAddressGenerator.md: there is no X term).
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure_dest_stride()
{
    using cfg::PackerReg;
    constexpr std::uint32_t y_stride = face_cols * pack_datum_bytes(P);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg0].Xstride, cfg::Sec::S0, 0>(),
        cfg::set<cfg::Packer[0].AddrCtrl[PackerReg::Reg0].Ystride, cfg::Sec::S0, y_stride>());
}

/// Configure, the Dest format the gasket and ReLU see (Dstacc, no override), and ReLU itself. ReluThreshold is not
/// written: only the threshold modes (ApplyRelu 2, 3) read it (ReLU.md), and this op sets mode 0 or 1. These are the
/// words shared with the math thread ("Shared config words and waits" in the file comment).
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure_dest_format_and_relu()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::Dstacc_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::Dstacc_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg2::Dstacc, cfg::Sec::S0, pack_in_data_format(P)>(), // FP16 for FP8 (E4M3) output, as the packer's
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0, P.relu ? pack_zero_relu : pack_no_relu>());
}

/// Configure, how the packer reads Dest: 32-bit datums with a 32-bit Dest or a 32-bit input format; 8-bit integers;
/// unsigned for UINT8 output; mantissa rounded to 10 bits where the conversion needs it (set_packer_config).
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure_dest_read()
{
    constexpr bool fp8_e4m3  = P.out_format == DataFormat::Fp8_e4m3;
    constexpr bool read_32b  = is_32bit_format(P.in_format) || P.dest_fp32;
    constexpr bool read_int8 = !read_32b && (P.in_format == DataFormat::Int8 || P.in_format == DataFormat::UInt8);
    constexpr bool round_10b = (P.dest_fp32 && P.in_format == DataFormat::Float16) || (!platform::format::has_5bit_exponent(P.in_format) && fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0, read_32b ? 1u : 0u>(),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0, P.out_format == DataFormat::UInt8 ? 1u : 0u>(),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0, read_int8 ? 1u : 0u>(),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0, round_10b ? 1u : 0u>());
}

/// Configure, packer 0's config (THCON SEC0 REG1), every field but L1_Dest_addr (execute's), Unp_LF8_4b_exp (the
/// unpacker's) and the word with Pack_L1_Acc (pack_configure_l1_acc). The exponent section: one exponent block per
/// face for block formats; none for FP8 and INT8.
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure_packer_words()
{
    using r1                           = cfg::ThconReg1Fields;
    constexpr std::uint32_t in_format  = pack_in_data_format(P);
    constexpr std::uint32_t out_format = format_id(P.out_format);
    constexpr bool no_exponents        = out_format == format_id(DataFormat::Lf8) || out_format == format_id(DataFormat::Int8);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r1::Row_start_section_size, cfg::Sec::S0, 0>(), // no row-start section: no compression
        cfg::set<r1::Exp_section_size, cfg::Sec::S0, no_exponents ? 0u : P.num_faces>());
    // The fields below without a comment are 0, as configure_pack (cpack_common.h) leaves them: it zeroes the whole
    // config before setting the formats and section sizes.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<r1::Disable_zero_compress, cfg::Sec::S0, 1>(),   // plain tiles, no zero compression
        cfg::set<r1::Add_l1_dest_addr_offset, cfg::Sec::S0, 0>(), // the address is L1_Dest_addr alone
        cfg::set<r1::Disable_pack_zero_flags, cfg::Sec::S0, 0>(),
        cfg::set<r1::ovrd_default_throttle_mode, cfg::Sec::S0, 0>(),
        cfg::set<r1::Out_data_format, cfg::Sec::S0, out_format>(),
        cfg::set<r1::In_data_format, cfg::Sec::S0, in_format>(),
        cfg::set<r1::Dis_shared_exp_assembler, cfg::Sec::S0, 0>(),
        cfg::set<r1::Auto_set_last_pacr_intf_sel, cfg::Sec::S0, 0>(),
        cfg::set<r1::Enable_out_fifo, cfg::Sec::S0, 0>(),
        cfg::set<r1::Sub_l1_tile_header_size, cfg::Sec::S0, 0>(), // 0: nothing to subtract; platform::l1::word_address already points one word before the data
        cfg::set<r1::Source_interface_selection, cfg::Sec::S0, 0>(), // read Dest, not L1
        cfg::set<r1::pack_start_intf_pos, cfg::Sec::S0, 0>(),
        cfg::set<r1::All_pack_disable_zero_compress_ovrd, cfg::Sec::S0, 0>(),
        cfg::set<r1::Add_tile_header_size, cfg::Sec::S0, 0>(),
        cfg::set<r1::pack_dis_y_pos_start_offset, cfg::Sec::S0, 0>(),
        cfg::set<r1::L1_source_addr, cfg::Sec::S0, 0>());
}

/// Configure, the word that holds Pack_L1_Acc: the bit on for Toggle::On and off otherwise. A Toggle::Runtime l1_acc
/// is execute's to turn on (pack_set_runtime_l1_acc).
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure_l1_acc()
{
    pack_write_l1_acc_word<P, P.l1_acc == Toggle::On ? 1u : 0u>();
}

/// Configure, edge masking off: one read per XY plane, every datum kept, every row on mask 0
/// (_llk_pack_reduce_mask_clear_). The other PACK_COUNTERS fields are 0, as configure_pack writes them (the whole
/// word 0, then pack_reads_per_xy_plane = 1): no automatic last-PACR generation.
inline __attribute__((always_inline)) void pack_configure_edge_mask()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, pack_one_read_per_plane>(),
        cfg::set<cfg::PackCounters::pack_xys_per_tile, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::pack_yz_transposed, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PackCounters::auto_ctxt_inc_xys_cnt, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, pack_all_datums_mask>(),
        cfg::set<cfg::PckEdgeMode::mode, cfg::Sec::S0, 0>(),
        cfg::set<cfg::PckEdgeTileRowSetSelect::select, cfg::Sec::S0, 0>());
    // Every 2-bit entry of TILE_ROW_SET_MAPPING_0 to 0, as immediate writes: a compiler that tracks Tensix state then
    // knows the value (not just "what GPR 0 held") and, when it removes a repeat, nothing is left behind (a WRCFG's
    // completion NOP stays as padding).
    pack_clear_row_set_mapping(std::make_index_sequence<cfg::detail::TileSetMappingEntryCount> {});
}

/// Configure, the two address modifiers the PACRs name.
inline __attribute__((always_inline)) void pack_configure_addr_mods()
{
    pack_set_addr_mod<pack_next_rows, pack_counter_step {.incr = pack_rows_per_pacr, .clr = 0}>();
    pack_set_addr_mod<pack_run_end, pack_counter_step {.incr = 0, .clr = 1}>();
}

/// Configure, the address counters to 0, except channel 1's X: the last datum of a row, so each read interface takes
/// one 16-datum row (InputNumDatums = channel 1 X - channel 0 X + 1).
inline __attribute__((always_inline)) void pack_configure_counters()
{
    using namespace hal::counters::adc;
    hal::counters::set<
        AddrCounter {.channel0 = {.x = 0, .y = 0, .z = 0, .w = 0}, .channel1 = {.x = face_cols - 1, .y = 0, .z = 0, .w = 0}}.for_clients<Client::Packers>()>();
}

/// Programs the packer for P, completely. Waits for earlier packs to finish first. The body is the order of what is
/// issued.
template <pack_params P>
inline __attribute__((always_inline)) void pack_configure()
{
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the MOP and strides assume four 16-row faces");
    static_assert(!P.untilize, "untilize is not ported yet");

    // Nothing below may change while an earlier pack still runs: hold config writes and address counter updates
    // until the packer is idle. This is the pack's one wait for the packer (execute relies on it). A plain STALLWAIT,
    // not a platform::stall_scope: a state-tracking compiler that is told the scope may delete the wait whenever it
    // finds the writes behind it not to need it, which by its model is always for the packer's own words (file comment).
    hal::sync::wait::stall<StallTarget::Config | StallTarget::Tdma, StallCondition::PackerIdle>();

    pack_configure_dest_stride<P>();          // 1. the Dest-side Y stride: one Dest row
    pack_configure_dest_format_and_relu<P>(); // 2. Dstacc and ReLU (the words shared with math)
    pack_configure_dest_read<P>();            // 3. how the packer reads Dest: width, sign, rounding
    pack_configure_packer_words<P>();         // 4. packer 0's config: formats, exponent section, no compression
    pack_configure_l1_acc<P>();               // 5. the word with Pack_L1_Acc
    pack_configure_edge_mask();               // 6. edge masking off
    pack_configure_addr_mods();               // 7. the two address modifiers the PACRs name
    pack_program_mop<P>();                    // 8. the MOP of one run; MopCfg is written by RISC-V stores behind a
                                              //    wait for the MOP expander (hal::mop::program)
    pack_configure_counters();                // 9. the address counters
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// Whether execute changes Pack_L1_Acc after configure set it: a sum_tiles pack whose first tile does not (or may
/// not) accumulate turns it on for the rest.
constexpr bool pack_sum_switches_l1_acc(const pack_params& params)
{
    return params.sum_tiles && params.l1_acc != Toggle::On && params.tiles > 1;
}

/// A Toggle::Runtime l1_acc: turns Pack_L1_Acc on when l1_acc is set (configure wrote it off). The packer is idle:
/// configure waited for it and nothing has been issued to it since. The bit is written as a constant, which a compiler
/// that tracks Tensix state can follow (a computed value would be a run-time instruction word).
template <pack_params P>
inline __attribute__((always_inline)) void pack_set_runtime_l1_acc(bool l1_acc)
{
    if constexpr (P.l1_acc == Toggle::Runtime)
    {
        if (l1_acc)
        {
            cfg::write<cfg::Access::TensixCfgUnit, cfg::ThconReg1Fields::Pack_L1_Acc, cfg::Sec::S0, 1>();
        }
    }
}

/// Points the packer's Dest read at Dest row dest_row, for the sum and the per-tile form: the GPR, the wait for it,
/// one WRCFG. Earlier packs read the old offset: configure waited for them (the packer is idle here; no second wait
/// for it, see the file comment). The wait is for the GPR.
/// The register is written whole from a GPR (Offset in the low bits, ZOffset 0): a byte-wise RMWCIB would read it
/// back first, which ttsim does not support for this register.
inline __attribute__((always_inline)) void pack_set_dest_row(std::uint32_t dest_row)
{
    hal::scalar::set(hal::gpr<platform::gpr_map::pack::scratch0>(), dest_row);
    hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<platform::gpr_map::pack::scratch0>());
}

/// Block form: the Dest row and the first tile's L1 address once, then one MOP run over every tile. first_tile is the
/// L1 address in 16-byte words.
template <pack_params P, class Tiles>
inline __attribute__((always_inline)) void pack_execute_block(std::uint32_t dest_row, std::uint32_t first_tile, const Tiles& tiles)
{
    using namespace hal::counters::adc;
    // The two run-time words, the Dest offset and the L1 address, go through GPRs: two SETDMAREG halves each, then
    // one WRCFG of the whole word (program_packer_destination; a byte-wise RMWCIB needs four run-time words for the
    // address, and reads the Dest offset register back, which ttsim does not support). Offset is the low field of
    // its word and ZOffset stays 0; L1_Dest_addr is a whole word.
    hal::scalar::set(hal::gpr<platform::gpr_map::pack::scratch0>(), dest_row);
    LLK_ASSERT(platform::l1::is_valid_word_address(first_tile + (wrapper::value(tiles) - 1) * P.tile_words), "the packed block ends outside L1");
    hal::scalar::set(hal::gpr<platform::gpr_map::pack::output_address>(), pack_l1_dest_addr(first_tile));

    // One STALLWAIT for the WRCFGs: until the scalar unit has written the GPRs. The packer is idle: configure waited
    // for it and issued nothing to it (earlier packs read the old Dest offset at every PACR). No second wait for
    // the packer here: see the file comment.
    hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0>(hal::gpr<platform::gpr_map::pack::scratch0>());
    // WrcfgCompletion::Wait: a NOP, then the SETADC below, keep the first PACR clear of the WRCFG.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<platform::gpr_map::pack::output_address>());

    // Y is 0: configure cleared it, and so does every run's closing PACR (pack_run_end). This SETADC (W is 0
    // already) is the second instruction between the WRCFG and the first PACR.
    hal::counters::set<AddrCounter {.channel0 = {.w = 0}}.for_clients<Client::Packers>()>();

    // One MOP run over every tile. A run-time count: the MOP is programmed here, with it (the MOP instruction
    // itself stays a constant word).
    if constexpr (wrapper::is_rt_v<Tiles>)
    {
        pack_program_runtime_mop<P>(wrapper::value(tiles));
    }
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

/// Sum form: every tile to the one L1 tile at first_tile (16-byte words). The first tile PACR by PACR, the switch of
/// Pack_L1_Acc, then one MOP run over the other tiles.
template <pack_params P, class Tiles>
inline __attribute__((always_inline)) void pack_execute_sum(std::uint32_t dest_row, std::uint32_t first_tile, const Tiles& tiles)
{
    pack_set_dest_row(dest_row);

    // The address once, for every tile: a tile's closing PACR (Last) makes the next PACR take its output address
    // from L1_Dest_addr again, and the L1 side's counters never move. The STALLWAIT keeps the PACRs from starting
    // before the address lands.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(pack_l1_dest_addr(first_tile));
    hal::sync::wait::stall<StallTarget::Pack, StallCondition::ConfigUnitIdle>();
    // The first tile, PACR by PACR: Y runs through its rows (and on through the rows of the tiles behind it).
    // Issued directly, so the MOP can hold the run over the other tiles and is programmed once.
#pragma GCC unroll 16
    for (std::uint32_t i = 0; i + 1 < pack_pacrs_per_tile(P); i++)
    {
        hal::pack::run<pack_transfer(pack_next_rows, hal::pack::TileBoundary::NotLast)>();
    }
    // The first tile's last PACR. A single tile (a compile-time count of one, or a run-time count that is one)
    // ends the run with it: Y back to 0.
    if (wrapper::value(tiles) == 1)
    {
        hal::pack::run<pack_transfer(pack_run_end, hal::pack::TileBoundary::Last)>();
    }
    else
    {
        hal::pack::run<pack_transfer(pack_next_rows, hal::pack::TileBoundary::Last)>();
        if constexpr (pack_sum_switches_l1_acc(P))
        {
            // The first tile went out as configure set Pack_L1_Acc; the rest add. The bit changes only once the
            // packer has finished the first tile, as reconfigure_packer_l1_acc (cpack_common.h) waits. The wait
            // holds the packer too, so that it survives a state-tracking compiler (file comment).
            hal::sync::wait::stall<StallTarget::Config | StallTarget::Pack, StallCondition::PackerIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, cfg::ThconReg1Fields::Pack_L1_Acc, cfg::Sec::S0, 1>();
            hal::sync::wait::stall<StallTarget::Pack, StallCondition::ConfigUnitIdle>();
        }
        // A run-time count: the MOP over the other tiles is programmed here, for them.
        if constexpr (wrapper::is_rt_v<Tiles>)
        {
            pack_program_runtime_mop<P>(wrapper::value(tiles) - 1);
        }
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run(); // every other tile; its final PACR clears Y
    }
}

/// Per-tile form: tile i of Dest to tile i of L1 from first_tile (16-byte words): Y, the L1 address and one MOP run
/// per tile.
template <pack_params P, class Tiles>
inline __attribute__((always_inline)) void pack_execute_per_tile(std::uint32_t dest_row, std::uint32_t first_tile, const Tiles& tiles)
{
    using namespace hal::counters::adc;
    pack_set_dest_row(dest_row);

    for (std::uint32_t tile = 0; tile < wrapper::value(tiles); tile++)
    {
        const std::uint32_t tile_address = first_tile + tile * P.tile_words;
        LLK_ASSERT(platform::l1::is_valid_word_address(tile_address), "pack address is outside L1");
        // Y to the Dest tile's first row; the tile's first PACR samples the L1 address (in-order behind the last
        // tile's PACRs, as _llk_pack_ relies on). The STALLWAIT keeps the PACRs from starting before the address lands.
        hal::counters::set(AddrCounter {.channel0 = {.y = tile * P.num_faces * P.face_r_dim}}.for_clients<Client::Packers>());
        cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0>(pack_l1_dest_addr(tile_address));
        hal::sync::wait::stall<StallTarget::Pack, StallCondition::ConfigUnitIdle>();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        // Y is back at 0 (pack_run_end); the L1 side's counters never moved.
    }
}

/// Packs every tile of the Dest tensor at Dest row dest_row (absolute) to the L1 tensor at byte address l1_address,
/// tile i of Dest (row-major) to tile i of L1. With P.sum_tiles, every tile to the one tile at l1_address: the first
/// as configure set L1 accumulation, the rest added into it. tiles is the tile count: a ct (P.tiles), or an
/// rt of 1..P.tiles. With a run-time count the block and the sum form program the MOP here
/// (pack_program_runtime_mop), where the count is known; configure then leaves it alone (pack_runtime_mop).
/// l1_acc is read only when P.l1_acc is Toggle::Runtime.
/// Call it directly behind pack_configure<P>: the packer is idle then, and execute does not wait for it again.
template <pack_params P, class Tiles, ctrt<Tiles, std::uint32_t> = 0>
inline __attribute__((always_inline)) void pack_execute(std::uint32_t dest_row, std::uint32_t l1_address, const Tiles& tiles, bool l1_acc = false)
{
    if constexpr (wrapper::is_rt_v<Tiles>)
    {
        LLK_ASSERT(wrapper::value(tiles) >= 1 && wrapper::value(tiles) <= P.tiles, "a run-time tile count is outside 1..its bound");
    }
    static_assert(wrapper::is_rt_v<Tiles> == P.runtime_tiles, "pack_configure<P> and pack_execute<P> must agree on whether the tile count is a run-time value");
    static_assert(cfg::DestTargetRegCfgPack::Offset.shamt(cfg::Sec::S0) == 0, "Offset is the low field of the word");
    const std::uint32_t first_tile = platform::l1::word_address(l1_address); // 16-byte words, one before the data (the tile header slot)
    LLK_ASSERT(dest_row < (1u << cfg::DestTargetRegCfgPack::Offset.width), "Dest row does not fit the packer's Dest offset");
    LLK_ASSERT(platform::l1::is_valid_word_address(first_tile), "pack address is outside L1");
    pack_set_runtime_l1_acc<P>(l1_acc);
    if constexpr (pack_in_blocks(P))
    {
        pack_execute_block<P>(dest_row, first_tile, tiles); // one address, one MOP run for every tile
    }
    else if constexpr (P.sum_tiles)
    {
        pack_execute_sum<P>(dest_row, first_tile, tiles); // one L1 tile: tile 0 by hand, the L1-acc switch, the MOP
    }
    else
    {
        pack_execute_per_tile<P>(dest_row, first_tile, tiles); // Y, the address and one MOP run per tile
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The pack thread's part of a Compute API pack of the Dest tensor at Dest row dest_row to the L1 tensor at byte
/// address l1_address: configure, then execute. l1_acc is read only when the Plan's l1_acc is Toggle::Runtime. counts
/// is the run-time side of the Plan's Work (an empty object when every count is compile time).
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void pack(const WorkCounts<Pl.work>& counts, std::uint32_t dest_row, std::uint32_t l1_address, bool l1_acc = false)
{
    static_assert(On == Thread::Pack, "the packer is driven from the pack thread");
    constexpr pack_params params = pack_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        pack_configure<params>();
        pack_execute<params>(dest_row, l1_address, plan_count<WorkCount::Tiles, Pl.work.tile_rows * Pl.work.tile_cols>(counts), l1_acc);
    }
}

} // namespace llk2::llk_lib
