// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, Blackhole: the parts every math op's configure shares. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/), except where noted. The op headers (matmul.h, eltwise_binary.h, reduce.h, copy.h,
// vec.h) call these from their <op>_configure and <op>_execute; nothing here runs on its own.
//
//   configure_math_formats<DestFp32, A, B>()   after math and the SFPU drain: Dest mode, the ALU format baseline,
//                                               implied SrcA format, Src bank release and the Src zero flag
//   counter_step, counter_stay / _clear / _to_cr / _next_fpu_op
//                                              how one row counter moves under an address modifier
//   set_addr_mod<Slot, A, B, D, Fidelity>()    programs one address modifier (ADDR_MOD) slot
//   reset_math_counters()                      every row counter (and its CR register) and the fidelity phase to 0
//   dest_rows_per_tile(params)                 Dest rows one tile of an op takes
//   dest_half_first_row(row)                   the first row of the Dest half a row is in: where ZEROACC counts faces from
//   set_math_dest_row(row)                     this thread's math Dest offset to an absolute Dest row
//   set_math_dest_tile<P>(dest_row, tile)      the same, to a tile of the tensor at dest_row
//   addr_mod_hold, set_hold_addr_mod()         the address modifier slot that moves no counter, and its programming
//   zero_dest_faces(row, first, faces, ...)    ZEROACC of faces of the tile at an absolute Dest row
//   run_tile_replay<Range, Phases>()           one tile from its replay slots, once per fidelity phase
//
// What an FPU op's configure owns, and what it shares. Every configure writes
// everything its op needs. Whether a second configure of the same op is REDUNDANT, so that a compiler that tracks
// Tensix state can remove it, depends on whether another op's configure wrote the same resource in between:
//   Dest mode and ALU formats   shared state config, the same values for every op of a kernel: redundant after the
//                               first. Only these need math and the SFPU drained; the STALLWAIT guards only them.
//   address modifiers           8 slots per thread. 7 moves nothing (addr_mod_hold): every op that needs such a
//                               slot programs and names that one, and SFPI's loads and stores name it too. 6 is
//                               the SFPU functions' stepping store. The FPU ops share 0..5: matmul needs 6,
//                               eltwise 4 (HiFi: 5), reduce 3, so they cannot have slots of their own. A change of
//                               op rewrites
//                               the slots whose values differ (thread config, latched in order with the
//                               instructions: no drain, as 1.0's reinit).
//   replay slots                32 per thread. matmul [16, 32), eltwise [8, 16), reduce [0, 4 x phases): an op kind's
//                               tile stays recorded while other kinds run, with one exception: SFPU functions on
//                               this thread record from slot 0 too (llk_lib/vec/: up to all 32 slots, the fast exp's
//                               body), so an SFPU call in between overwrites the FPU ops' tiles, and their next
//                               configure records them again.
//                               Nothing relies on a recording made by an earlier configure: every configure records.
//   MOP                         one MopCfg per thread, shared by both templates (9 words, written by the RISC-V
//                               after a sync with the MOP expander). No FPU op uses it: a tile is its replay played
//                               once per fidelity phase, as 1.0's matmul_block_no_mop, and (HiFi) the Src release.
//
// Address modifiers move the math row counters (RWCs) after each FPU instruction that names them. The counters
// follow the ISA's ApplyAddrMod (tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/RWCs.md): a plain
// step adds incr to the counter; a carriage-return (CR) step adds incr to the counter's CR register and copies it to
// the counter; clear zeroes both, whatever the CR bit says.
//
// Not HAL: DEST_REGISTER_HALF_SIZE (tensix_types.h). The Src zero-flag rule (src_zero_flag_disabled,
// llk_lib/formats.h) is llk-lib's own. The 1.0 LLK's caches of the flag (math::src_zero_flag_*) are not written: they
// are that library's own state, and only its functions read them.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/formats.h"
#include "llk_lib/platform/format.h"
#include "llk_lib/platform/issue.h"
#include "tensix_types.h"  // DataFormat, DEST_REGISTER_HALF_SIZE
#include "tensor/tensor.h" // max_face_rows

namespace llk2::llk_lib
{

/// Rows of SrcB and Dest one MVMUL covers, and rows of SrcA, SrcB and Dest one ELWADD, ELWSUB or MOVA2D (its
/// eight-row form) covers.
inline constexpr std::uint32_t rows_per_fpu_op = 8;

/// Replay slots of one math-thread tile, by FPU op kind (see the top of this file).
inline constexpr std::uint32_t matmul_replay_start  = 16; ///< [16, 32): matmul's 16 MVMULs
inline constexpr std::uint32_t eltwise_replay_start = 8;  ///< [8, 16): an eltwise tile's 8 instructions
inline constexpr std::uint32_t reduce_replay_start  = 0;  ///< [0, 4 x phases): a reduce tile's pools, at most 16

/// Row counter behavior of one address modifier: {increment, clear, carriage return} for SrcA, SrcB or Dest.
struct counter_step
{
    std::uint32_t incr; ///< rows the counter (or its CR register, when cr is set) moves on
    std::uint32_t clr;  ///< 1: counter and CR register to 0; cr is then ignored (ApplyAddrMod)
    std::uint32_t cr;   ///< 1: add incr to the CR register and copy it to the counter
};

/// The counter does not move.
inline constexpr counter_step counter_stay {0, 0, 0};

/// The counter and its CR register to 0. The CR bit is left 0: clear ignores it.
inline constexpr counter_step counter_clear {0, 1, 0};

/// The counter back to its CR register (the CR register stays).
inline constexpr counter_step counter_to_cr {0, 0, 1};

/// The counter on by the rows of one FPU instruction (rows_per_fpu_op).
inline constexpr counter_step counter_next_fpu_op {rows_per_fpu_op, 0, 0};

/// The largest row step an address modifier can hold for Dest (ADDR_MOD_DST_SEC*_DestIncr, 10 bits).
inline constexpr std::uint32_t max_dest_counter_step = 0x3ff;

/// Dest rows one tile of an op takes: faces per tile times rows per face. P is any math params struct with
/// num_faces and face_r_dim.
template <class P>
constexpr std::uint32_t dest_rows_per_tile(const P& params)
{
    return params.num_faces * params.face_r_dim;
}

/// Rows in one half of Dest. A ZEROACC of a face (hal::dst::ZeroScope::Face) counts 16-row faces from the start of the
/// Dest half the math Dest offset selects, and Blackhole picks that half from the offset plus the index
/// (tt-isa-documentation, ZEROACC.md). So an op that zeroes faces of a tile at an absolute Dest row
/// first moves the offset to the start of the row's half (dest_half_first_row) and keeps the index below the half's size.
/// The same with a 32-bit Dest: Adj32 keeps row bit 9 (Dst.md), so the upper half starts at row 512.
inline constexpr std::uint32_t dest_half_rows = DEST_REGISTER_HALF_SIZE;

/// The first row of the Dest half that holds Dest row `row` (absolute).
constexpr std::uint32_t dest_half_first_row(std::uint32_t row)
{
    return row - row % dest_half_rows;
}

/// Points this thread's math Dest offset at Dest row `row` (absolute): the FPU's and the SFPU's Dest addresses on this
/// thread count from there.
inline __attribute__((always_inline)) void set_math_dest_row(std::uint32_t row)
{
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DestTargetRegCfgMath::Offset, hal::cfg::Sec::S0>(row);
}

/// Points this thread's math Dest offset at tile `tile` of the tensor that starts at Dest row dest_row (P's tile size).
template <auto P>
inline __attribute__((always_inline)) void set_math_dest_tile(std::uint32_t dest_row, std::uint32_t tile)
{
    set_math_dest_row(dest_row + tile * dest_rows_per_tile(P));
}

/// Programs address modifier SlotNumber: the number an instruction names the slot by, 0 to 7 (an op's slot constants).
/// Fields this does not name are written 0, as addr_mod_t does. FidelityClear 1: the fidelity phase goes back to 0
/// (FidelityIncr is then 0).
template <std::uint8_t SlotNumber, counter_step A, counter_step B, counter_step D, std::uint32_t FidelityIncr = 0, std::uint32_t FidelityClear = 0>
inline __attribute__((always_inline)) void set_addr_mod()
{
    namespace cfg = hal::cfg;
    using cfg::AddrMod;
    static_assert(SlotNumber < 8, "there are 8 address modifier slots");
    constexpr cfg::Sec Slot = static_cast<cfg::Sec>(SlotNumber); // the HAL names the slot's config section
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::SrcA].Incr, Slot, A.incr>(),
        cfg::set<AddrMod[cfg::SrcA].Clear, Slot, A.clr>(),
        cfg::set<AddrMod[cfg::SrcA].CR, Slot, A.cr>(),
        cfg::set<AddrMod[cfg::SrcB].Incr, Slot, B.incr>(),
        cfg::set<AddrMod[cfg::SrcB].Clear, Slot, B.clr>(),
        cfg::set<AddrMod[cfg::SrcB].CR, Slot, B.cr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::Dest].Incr, Slot, D.incr>(),
        cfg::set<AddrMod[cfg::Dest].Clear, Slot, D.clr>(),
        cfg::set<AddrMod[cfg::Dest].CR, Slot, D.cr>(),
        // The fidelity phase moves on (HiFi), stays, or goes back to 0.
        cfg::set<AddrMod[cfg::Fidelity].Incr, Slot, FidelityIncr>(),
        cfg::set<AddrMod[cfg::Fidelity].Clear, Slot, FidelityClear>());
    // No bias counter step.
    cfg::write<cfg::Access::TensixCfgUnit, AddrMod[cfg::Bias].Incr, Slot, 0>();
}

/// The address modifier slot that moves no counter, on every thread that drives the FPU or the SFPU: what a ZEROACC, a
/// move or an SFPU load names when the instruction addresses its rows itself. It is the slot SFPI's loads and stores
/// name (sfpi_constants.h: SFPLOAD_ADDR_MODE_NOINC), so the FPU ops and the SFPU functions agree on it.
inline constexpr std::uint8_t addr_mod_hold = 7;

/// Programs addr_mod_hold. Every op that names the slot calls this in its configure.
inline __attribute__((always_inline)) void set_hold_addr_mod()
{
    set_addr_mod<addr_mod_hold, counter_stay, counter_stay, counter_stay>();
}

/// Every math row counter (SrcA, SrcB, Dest), their CR registers, and the fidelity phase to 0.
inline __attribute__((always_inline)) void reset_math_counters()
{
    using namespace hal::counters::rwc;
    set<MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .clear_fidelity = true}>();
}

/// Dest mode, ALU format baseline, implied SrcA format, Src bank release and Src zero flag for an FPU op whose SrcA
/// holds SrcAFormat and SrcB SrcBFormat (the register formats). The math side of 1.0's hw_configure. The writes
/// wait for math and the SFPU to drain; the wait is theirs alone (platform::stall_scope).
template <bool DestFp32, DataFormat SrcAFormat, DataFormat SrcBFormat>
inline __attribute__((always_inline)) void configure_math_formats()
{
    using hal::sync::StallCondition;
    using hal::sync::StallTarget;
    const auto writes = [] __attribute__((always_inline))
    {
        namespace cfg = hal::cfg;
        // ZEROACC indices are relative to the Dest offset, not absolute.
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
        constexpr std::uint32_t int8_math = platform::format::uses_integer_math(SrcAFormat) ? 1u : 0u;
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, int8_math>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, DestFp32 ? 1u : 0u>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, DestFp32 ? 1u : 0u>();
        // Src formats come from what the unpackers wrote (Blackhole's implied formats), not an override.
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
            cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
            cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
            cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());
        // SrcA's format is the one implied by the bank the unpacker filled, and a Src release (a flip or an
        // instruction's release) hands the bank back to the unpacker: it clears its data valid. Each is one whole
        // thread-config word.
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

        // Src zero-flag baseline for these formats.
        constexpr std::uint32_t zero_flag = src_zero_flag_disabled(SrcAFormat, SrcBFormat) ? 1u : 0u;
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, zero_flag>();
    };
    platform::stall_scope<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>(writes);
}

/// ZEROACC of `faces` 16-row faces, from face `first_face` of the tile that starts at Dest row tile_row (absolute):
/// marks them zero, or with ZeroFlagAction::ClearFlags marks them written. The math Dest offset moves to the start of
/// the tile's Dest half (dest_half_first_row) and is left there: the caller points it at its tile next. The ZEROACCs
/// name addr_mod_hold (their 16-row form applies an address modifier), so the op's configure must have programmed it
/// and the Dest row counter must be 0.
inline __attribute__((always_inline)) void zero_dest_faces(
    std::uint32_t tile_row,
    std::uint32_t first_face,
    std::uint32_t faces,
    hal::dst::DestWidth width,
    hal::dst::ZeroFlagAction flags = hal::dst::ZeroFlagAction::MarkUndefined)
{
    const std::uint32_t half_row = dest_half_first_row(tile_row);
    set_math_dest_row(half_row);
    const std::uint32_t first = (tile_row - half_row) / max_face_rows + first_face;
    for (std::uint32_t face = 0; face < faces; face++)
    {
        hal::dst::run(
            hal::dst::Zero {.scope = hal::dst::ZeroScope::Face, .index = first + face, .flags = flags, .width = width, .address_mode = addr_mod_hold});
    }
}

/// One tile of an FPU op: its replay slots Range played once per fidelity phase, each play issued directly (no MOP:
/// MopCfg is one register set per thread, so two op kinds could never both keep a MOP program).
template <hal::replay::BufferRange Range, std::uint32_t Phases>
inline __attribute__((always_inline)) void run_tile_replay()
{
#pragma GCC unroll 4
    for (std::uint32_t phase = 0; phase < Phases; phase++)
    {
        hal::replay::run<Range>();
    }
}

} // namespace llk2::llk_lib
