// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, reduce on Blackhole: a row max or row sum. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/), except where noted. The FPU's part of a Compute API reduce; the unpackers' part is
// unpack.h (resolver/reduce.h has the walk). The shape of an op: llk_lib/thread.h.
//
//   reduce_params              one struct; reduce_configure and reduce_execute take the same one
//   reduce_params_from(Plan)   fills it from the Resolver's Plan
//   reduce_configure<P>()      Dest mode and ALU formats, the address modifiers, the replay of one input tile's pools
//                              (no MOP: execute plays the replay itself, math_common.h)
//   reduce_execute<P>(dest_row, folded)
//                              every output tile, from Dest row dest_row: pool its tile row of inputs, then transpose
//                              the result into column 0. folded is the input tiles of each output tile: a ct or, for
//                              a row length known only at run time, an rt (resolver/work_counts.h)
//   reduce_product_execute<P>(dest_row)
//                              after the product of a sum in a 32-bit Dest: clears what it left outside column 0
//   reduce<On, Plan>(counts, dest_row)
//                              the per-thread call the Compute API makes: configures and executes, only when this
//                              build is for thread On (the math thread). counts (WorkCounts) is the run-time side of the
//                              Plan's Work
//
// How one output tile is computed. The unpacker writes each input tile to a SrcA bank with every face transposed, so
// SrcA column j of face f is row j of the input's face f. A pool (GMPOOL for max, GAPOOL for sum) folds the 16 rows of
// one SrcA face into one 16-datum Dest row, combining it with what that row holds (max, or +=):
//   faces 0, 1 (tile rows 0..15)   -> Dest row 0 of the output tile
//   faces 2, 3 (tile rows 16..31)  -> Dest row reduce_bottom_row (16: face 1's row 0)
// so after all N input tiles of the tile row, Dest row 0 datum j is the reduction of tile row j, and row 16 datum j
// that of tile row 16 + j. GMPOOL takes each datum times 2^(exponent of the scaler, SrcB row 0), GAPOOL is
// SrcB row 0 times the face: with a 1.0 scaler, a plain max or sum. The scaler's bank is held for the whole tile row
// and released after it; each data bank is released by the tile's last pool.
//
// Then each pooled row becomes column 0 of its face: Dest row -> SrcB row 16 (MOVD2B), TRNSPSRCB (transposes SrcB
// rows 16..31), SrcB rows 16..31 -> Dest rows (MOVB2D, 4 x 4 rows):
//   Dest row 0  -> face 0 column 0 (tile rows 0..15)
//   Dest row 16 -> face 2 column 0 (tile rows 16..31)
// Before each MOVD2B, ZEROSRC clears the SrcB bank (the scaler is no longer needed), so the other 15 columns come out
// 0; faces 1 and 3 are marked zero (ZEROACC) after. The tile is then column 0 and every other datum 0, so pack needs
// no reduce mask. The moves run with the Src zero flag disabled, so a datum whose low byte is 0 is not flushed to 0
// on the way.
//
// A 32-bit Dest. The pools write FP32 rows. The moves take a datum in the format of SrcA, which is the scaler bank's
// implied one (BF16 for a BF16 scaler: 16 bits of the 32), so around them SrcA's format is set to TF32 from the format
// register: MOVD2B then takes a datum's top 19 bits and MOVB2D writes them back as FP32, the low 13 bits 0. That is
// every bit of a max (GMPOOL's operands are Src datums, 19 bits at most). A sum would be cut to 19 bits (11
// significant ones), towards zero: the low 16 bits cannot follow in a second trip (with a 32-bit Dest, MOVB2D's
// low-half form overwrites the whole datum; measured on the board: every result reads 0, in either order of the two
// trips, though _llk_math_reduce_block_max_row_ issues that sequence for FP32).
//
// So a sum in a 32-bit Dest is not pooled (P.by_product; resolver/reduce.h): the Plan is matmul's, in times the
// scaler tile transposed, and matmul (llk_lib/matmul.h) runs it: MVMUL adds the products up in Dest at its full
// width, in column 0 directly. A seed is acc's contents as they are (matmul adds into Dest). The scaler's transpose
// has ones in column 16 too (face 2's row 0), so the product leaves the same sums there: reduce_product_execute marks
// faces 1 and 3 zero, as the pooled form does.
//
// Seeds (P.accumulate). Without one, the two pooled rows are first marked undefined (ZEROACC), which a pool reads as its identity:
// -infinity for GMPOOL, 0 for GAPOOL. With one, acc already holds a previous result (column 0 of faces 0 and 2) moved
// into the pooled rows: a copy<copy_modifiers{.transpose = true}> of it puts face 0's column 0 in Dest row 0 and face
// 2's in row 16 (face 2 -> face 1), exactly where the pools fold.
//
// Ported from: _llk_math_reduce_block_max_row_ (experimental/llk_math_reduce_custom.h: the pools, the transpose
// through SrcB, _configure_preserve_zero_flag_state_ and, with a 32-bit Dest, the TF32 SrcA format around it), the sequence SDPA 1.0's
// reduce_block_max_row_runtime runs, extended to GAPOOL for the sum. Differs from it in one place: SDPA 1.0 seeds
// with copy_tile_custom (a within-face transpose), which puts face 2's column into row 32; here the bottom face row
// pools into row 16, to take the plain transposed copy.
//
// Departures from the shape (llk_lib/thread.h), and why:
//   - execute writes ALU config per output tile: the Src zero flag off for the moves and back, and with a 32-bit
//     Dest SrcA's format (TF32 from the format register) for the moves and back to the implied one. The pools and
//     the moves of one tile need different values, so the writes are inside the call's work. They follow the last
//     pool and the last move with no wait for the FPU ("Pooled rows to column 0" in reduce_execute has the reason
//     and the test). SrcA's format register is left holding TF32: it is read only while the implied format is off,
//     and every math configure turns the implied format on (configure_math_formats).
//   - a sum in a 32-bit Dest has no configure of its own: matmul's configure and execute run its Plan.
//
// Not HAL: nothing.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/src.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h" // configure_math_formats, set_addr_mod, dest_half_first_row, the replay slots
#include "llk_lib/matmul.h"      // a sum in a 32-bit Dest is matmul's Plan
#include "llk_lib/thread.h"
#include "resolver/plan.h"
#include "resolver/reduce_function.h"
#include "resolver/work_counts.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;

// ---- parameters -----------------------------------------------------------------------------------------------------

/// Parameters of reduce on the math thread. configure and execute take the same struct.
struct reduce_params
{
    std::uint32_t face_r_dim;    ///< rows per face
    std::uint32_t num_faces;     ///< faces per tile
    std::uint32_t tile_rows;     ///< output tiles (one per tile row of the input)
    std::uint32_t k_tiles;       ///< input tiles folded into each output tile (the tile row's length)
    ReduceKind kind;             ///< max (GMPOOL) or sum (GAPOOL)
    bool accumulate;             ///< acc holds a seed in the pooled rows; do not clear them first
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi, 2..4 HiFi2..4 (sum only; max has one phase)
    bool dest_fp32;              ///< 32-bit Dest mode
    DataFormat src_a_format;     ///< the data's register format
    DataFormat src_b_format;     ///< the scaler's register format
    bool by_product;             ///< a sum in a 32-bit Dest: matmul runs the Plan, and this file only clears faces 1 and 3
};

/// The Plan of a reduce as math parameters.
constexpr reduce_params reduce_params_from(const Plan& plan)
{
    return {
        .face_r_dim    = plan.work.face_r_dim,
        .num_faces     = plan.work.num_faces,
        .tile_rows     = plan.work.tile_rows,
        .k_tiles       = plan.work.k_tiles,
        .kind          = plan.math.reduce,
        .accumulate    = plan.math.accumulate,
        .math_fidelity = plan.math.math_fidelity,
        .dest_fp32     = plan.math.dest_fp32,
        .src_a_format  = plan.in0.writes,
        .src_b_format  = plan.in1.writes,
        .by_product    = plan.math.by_product,
    };
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

inline constexpr std::uint32_t reduce_top_row            = 0;             ///< the Dest row faces 0 and 1 pool into
inline constexpr std::uint32_t reduce_bottom_row         = max_face_rows; ///< the Dest row faces 2 and 3 pool into (face 1's row 0)
inline constexpr std::uint32_t reduce_srcb_transpose_row = 16;            ///< TRNSPSRCB transposes SrcB rows 16..31 (the instruction's own rows)
inline constexpr std::uint32_t reduce_rows_per_movb2d    = 4;             ///< MOVB2D's four-row form
inline constexpr std::uint32_t reduce_replay_slots       = matmul_replay_start - reduce_replay_start; ///< 16: what is below matmul's tile

// The address modifier slots the replay and the moves use.
inline constexpr std::uint8_t reduce_next_phase = 0; ///< nothing moves but the fidelity phase: the next phase of a sum's pool
inline constexpr std::uint8_t reduce_next_face  = 1; ///< SrcA on by one face; the fidelity phase back to 0
inline constexpr std::uint8_t reduce_tile_end   = 2; ///< SrcA back to row 0; the fidelity phase back to 0

// The moves and ZEROACCs address their rows themselves and name addr_mod_hold (math_common.h). ZEROACC applies its
// address modifier too: with reduce_next_phase it would leave the next tile's pools on a later fidelity phase (a sum
// of 0).

/// The width a ZEROACC of P's Dest rows takes.
constexpr hal::dst::DestWidth reduce_dest_width(const reduce_params& params)
{
    return params.dest_fp32 ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
}

/// Fidelity phases one pool runs: the kernel's for a sum (LoFi: one), one for max (no multiply).
constexpr std::uint32_t reduce_phases(const reduce_params& params)
{
    return params.kind == ReduceKind::Sum && params.math_fidelity != 0 ? params.math_fidelity : 1;
}

/// The fidelity step after the last phase of a pool: the 2-bit phase counter wraps back to 0 (RWCs.md: uint2_t).
constexpr std::uint32_t reduce_phase_wrap(const reduce_params& params)
{
    constexpr std::uint32_t phase_count = 4; // the fidelity phase counter is 2 bits
    return (phase_count - (reduce_phases(params) - 1)) % phase_count;
}

/// Replay slots one input tile takes: one pool per face and phase.
constexpr std::uint32_t reduce_ops_per_tile(const reduce_params& params)
{
    return params.num_faces * reduce_phases(params);
}

/// The replay slots of one input tile: its pools.
constexpr hal::replay::BufferRange reduce_tile_replay(const reduce_params& params)
{
    return {reduce_replay_start, reduce_ops_per_tile(params)};
}

/// The Dest row (from the output tile's first row) face f pools into.
constexpr std::uint32_t reduce_pool_row(std::uint32_t face)
{
    return face < faces_per_face_row ? reduce_top_row : reduce_bottom_row;
}

/// One pool of P into Dest row Row, then the counters move by address modifier AddrMod; Release hands a Src bank back.
template <reduce_params P, std::uint32_t Row, std::uint8_t AddrMod, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void reduce_pool()
{
    constexpr auto function = P.kind == ReduceKind::Max ? hal::fpu::PoolFunction::Maximum : hal::fpu::PoolFunction::Sum;
    hal::fpu::run<hal::fpu::Pool {.function = function, .address_modifier = AddrMod, .dest_row_offset = Row, .release = Release}>();
}

/// Face F of one input tile: every fidelity phase, then on to the next face (the last face releases the data bank).
template <reduce_params P, std::uint32_t F>
inline __attribute__((always_inline)) void reduce_pool_face()
{
    for (std::uint32_t phase = 0; phase + 1 < reduce_phases(P); phase++)
    {
        reduce_pool<P, reduce_pool_row(F), reduce_next_phase>();
    }
    if constexpr (F + 1 < P.num_faces)
    {
        reduce_pool<P, reduce_pool_row(F), reduce_next_face>();
    }
    else
    {
        reduce_pool<P, reduce_pool_row(F), reduce_tile_end, hal::fpu::SourceRelease::SrcA>();
    }
}

// ---- configure ------------------------------------------------------------------------------------------------------

/// Programs the math thread for P: Dest mode and ALU formats (math_common.h), the address modifiers, the replay of
/// one input tile.
template <reduce_params P>
inline __attribute__((always_inline)) void reduce_configure()
{
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the pooled rows and moves assume four 16-row faces");
    static_assert(P.kind == ReduceKind::Max || P.kind == ReduceKind::Sum, "reduce is max or sum");
    static_assert(reduce_ops_per_tile(P) <= reduce_replay_slots, "one input tile's pools must fit reduce's 16 replay slots");

    configure_math_formats<P.dest_fp32, P.src_a_format, P.src_b_format>();

    // Address modifiers (counter_step: math_common.h). The pools read SrcA from the counter (one face per 16 rows) and
    // SrcB row 0; Dest rows come from the instructions, so only SrcA and the fidelity phase move.
    constexpr counter_step next_face {.incr = max_face_rows};
    set_addr_mod<reduce_next_phase, counter_stay, counter_stay, counter_stay, 1 /* FidelityIncr */>();
    set_addr_mod<reduce_next_face, next_face, counter_stay, counter_stay, reduce_phase_wrap(P)>();
    set_addr_mod<reduce_tile_end, counter_clear, counter_stay, counter_stay, reduce_phase_wrap(P)>();
    set_hold_addr_mod();

    // One input tile: faces 0, 1 into the top row, 2, 3 into the bottom row; the last pool releases the data bank.
    constexpr hal::replay::BufferRange tile_replay = reduce_tile_replay(P);
    hal::replay::record<tile_replay>(
        []
        {
            reduce_pool_face<P, 0>();
            reduce_pool_face<P, 1>();
            reduce_pool_face<P, 2>();
            reduce_pool_face<P, 3>();
        });

    reset_math_counters();
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// Writes the Src zero-flag disable bit (1: keep every datum on a move).
template <std::uint32_t Disabled>
inline __attribute__((always_inline)) void reduce_set_src_zero_flag()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, Disabled>();
}

/// One MOVB2D of the transposed SrcB rows back to Dest: four rows, the Quarter-th four of face Face's 16.
template <std::uint32_t Face, std::uint32_t Quarter>
inline __attribute__((always_inline)) void reduce_rows_back()
{
    using hal::move::Dst;
    using hal::move::SrcB;
    hal::move::run<hal::move::Transfer<SrcB, Dst> {
        .source_row      = reduce_srcb_transpose_row + Quarter * reduce_rows_per_movb2d,
        .destination_row = Face * max_face_rows + Quarter * reduce_rows_per_movb2d,
        .address_mode    = addr_mod_hold,
        .number_of_rows  = hal::move::rows::Four}>();
}

/// Pooled Dest row Row (from the output tile's first row) to column 0 of face Face: through SrcB rows 16..31,
/// transposed. The SrcB bank is cleared first, so the face's other columns come out 0.
template <std::uint32_t Row, std::uint32_t Face>
inline __attribute__((always_inline)) void reduce_row_to_column()
{
    using hal::move::Dst;
    using hal::move::SrcB;
    static_assert(max_face_rows == 4 * reduce_rows_per_movb2d, "a face goes back in four MOVB2Ds of four rows");
    hal::src<hal::SrcB>.fill<hal::src_ops::BankScope::Current, hal::src_ops::BankOwner::Math, hal::src_ops::Fill::Zero>();
    hal::move::run<hal::move::Transfer<Dst, SrcB> {
        .source_row = Row, .destination_row = reduce_srcb_transpose_row, .address_mode = addr_mod_hold, .number_of_rows = hal::move::rows::One}>();
    hal::src<hal::SrcB>.transpose();
    reduce_rows_back<Face, 0>();
    reduce_rows_back<Face, 1>();
    reduce_rows_back<Face, 2>();
    reduce_rows_back<Face, 3>();
}

/// Which format the moves between Dest and SrcB take a datum in. Tf32 1: TF32, from SrcA's format register, so a
/// 32-bit Dest datum's top 19 bits make the trip. Tf32 0: the format implied by the bank the unpacker filled, as
/// configure_math_formats leaves it. SrcA's format register then still holds TF32: it is read only while the implied
/// format is disabled, which nothing but this function does, and it writes the register then.
template <std::uint32_t Tf32>
inline __attribute__((always_inline)) void reduce_set_moves_tf32()
{
    if constexpr (Tf32 != 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg0::SrcA, cfg::Sec::S0, format_id(DataFormat::Tf32)>();
    }
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, Tf32>();
}

/// Marks Dest face `face` of the output tile at Dest row tile_row (absolute) zero (zero_dest_faces, math_common.h),
/// and puts the math Dest offset back on the tile.
template <reduce_params P>
inline __attribute__((always_inline)) void reduce_zero_face(std::uint32_t tile_row, std::uint32_t face)
{
    zero_dest_faces(tile_row, face, 1 /*faces*/, reduce_dest_width(P));
    set_math_dest_row(tile_row);
}

/// Every output tile: pool its input tiles, then move the pooled rows into column 0. dest_row is the absolute Dest
/// row the acc tensor starts at (compute_api/dest.h). folded: the input tiles of each output tile, a ct
/// (P.k_tiles) or, for a row length known only at run time, an rt of at most that (resolver/work_counts.h).
template <reduce_params P, class Folded, ctrt<Folded, std::uint32_t> = 0>
inline __attribute__((always_inline)) void reduce_execute(std::uint32_t dest_row, const Folded& folded)
{
    using namespace hal::counters::rwc;
    // The Src zero-flag rule, as configure_math_formats applies it.
    constexpr std::uint32_t default_zero_flag = src_zero_flag_disabled(P.src_a_format, P.src_b_format) ? 1u : 0u;
    constexpr std::uint32_t keep_every_datum = 1;
    constexpr MathCounter release_scaler {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = Src::B, .clear_fidelity = true};
    if constexpr (wrapper::is_rt_v<Folded>)
    {
        LLK_ASSERT(wrapper::value(folded) >= 1 && wrapper::value(folded) <= P.k_tiles, "a run-time row length is outside 1..its bound");
    }

    for (std::uint32_t tile = 0; tile < P.tile_rows; tile++)
    {
        const std::uint32_t tile_row = dest_row + tile * dest_rows_per_tile(P);
        set_math_dest_row(tile_row);
        if constexpr (!P.accumulate)
        {
            // The pools' identity in the two rows they fold into (-inf for max, 0 for sum).
            hal::dst::run<hal::dst::Zero {
                .scope = hal::dst::ZeroScope::SingleRow, .index = reduce_top_row, .width = reduce_dest_width(P), .address_mode = addr_mod_hold}>();
            hal::dst::run<hal::dst::Zero {
                .scope = hal::dst::ZeroScope::SingleRow, .index = reduce_bottom_row, .width = reduce_dest_width(P), .address_mode = addr_mod_hold}>();
        }
        for (std::uint32_t k = 0; k < wrapper::value(folded); k++)
        {
            // One input tile: every pool of every fidelity phase is in the replay, so it plays once.
            hal::replay::run<reduce_tile_replay(P)>();
        }

        // Pooled rows to column 0, then the scaler's bank back to the unpacker and faces 1, 3 to zero.
        // The two flag writes follow the last pool and the last MOVB2D with no wait for the FPU. Who reads the flag
        // (tt-isa-documentation, the functional models): MOVB2D does; GMPOOL and MOVD2B do not; GAPOOL's model is
        // MVMUL's, which does. No wait is needed: an FPU instruction takes its config when it leaves the Wait Gate
        // (the SFPI compiler's instruction table: "latches at the math pop"), and with reduce's formats
        // (8-bit-exponent floats) the flag changes no result. Measured on the board
        // (tests/python_tests/test_llk2_reduce_zero_flag_probe.py): this sequence packs the same bits as one that
        // waits for the FPU in front of each write, and as one with the flag set through the pools or clear through
        // the moves.
        reduce_set_src_zero_flag<keep_every_datum>();
        if constexpr (P.dest_fp32)
        {
            reduce_set_moves_tf32<1>();
        }
        reduce_row_to_column<reduce_top_row, 0>();
        reduce_row_to_column<reduce_bottom_row, 2>();
        if constexpr (P.dest_fp32)
        {
            reduce_set_moves_tf32<0>();
        }
        reduce_set_src_zero_flag<default_zero_flag>();
        set<release_scaler>();
        reduce_zero_face<P>(tile_row, 1);
        reduce_zero_face<P>(tile_row, 3);
    }
}

/// After the product of a sum by product (matmul's, into the acc tensor at absolute Dest row dest_row): faces 1 and 3
/// of every output tile to zero, which leaves column 0 and every other datum 0.
template <reduce_params P>
inline __attribute__((always_inline)) void reduce_product_execute(std::uint32_t dest_row)
{
    set_hold_addr_mod();
    for (std::uint32_t tile = 0; tile < P.tile_rows; tile++)
    {
        const std::uint32_t tile_row = dest_row + tile * dest_rows_per_tile(P);
        reduce_zero_face<P>(tile_row, 1);
        reduce_zero_face<P>(tile_row, 3);
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The math thread's part of a Compute API reduce: configure, then every output tile into Dest from dest_row. counts
/// is the run-time side of the Plan's Work (an empty object when every count is compile time).
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void reduce(const WorkCounts<Pl.work>& counts, std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    constexpr reduce_params params = reduce_params_from(Pl);
    if constexpr (params.by_product)
    {
        matmul<On, Pl>(counts, dest_row);
        if constexpr (runs_on<On>)
        {
            reduce_product_execute<params>(dest_row);
        }
    }
    else if constexpr (runs_on<On>)
    {
        reduce_configure<params>();
        reduce_execute<params>(dest_row, plan_count<WorkCount::K, Pl.work.k_tiles>(counts));
    }
}

} // namespace llk2::llk_lib
