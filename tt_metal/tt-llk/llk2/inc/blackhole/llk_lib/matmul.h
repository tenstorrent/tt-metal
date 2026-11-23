// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, matmul on Blackhole. Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/). The FPU's
// part of a Compute API matmul; the unpackers' part is unpack.h. The shape of an op: llk_lib/thread.h.
//
//   matmul_params                one struct; matmul_configure and matmul_execute take the same one
//   matmul_params_from(Plan)     fills it from the Resolver's Plan
//   matmul_configure<P>()        Dest mode and ALU formats, the address modifiers, the replay of one tile's MVMULs
//                                (no MOP: execute plays the replay once per fidelity phase, math_common.h)
//   matmul_execute<P>(dest_row, k_steps, streamed)
//                                every K step: every output tile, from Dest row dest_row, gets in0's tile k (SrcB)
//                                times in1's tile k (SrcA) added to it in Dest. The two counts are a ct, or an rt for
//                                a K or an N known only at run time (resolver/work_counts.h)
//   matmul<On, Plan>(counts, dest_row)
//                                the per-thread call the Compute API makes: configures and executes, only when this
//                                build is for thread On (the math thread). counts (WorkCounts) is the run-time side of
//                                the Plan's Work
//
// How one tile is computed. An MVMUL multiplies 8 rows of SrcB (8 x 16 datums of an in0 face) by a 16 x 16 SrcA
// face (in1) and adds the 8 x 16 product into 8 rows of Dest. A 32x32 output tile has 4 faces of 16 rows, and
// each needs two products along K (in0's two K faces times in1's), so a tile is 16 MVMULs. They run from a replay
// buffer; between them, the address modifiers (ADDR_MODs) move the SrcA, SrcB and Dest row counters:
//   matmul_next_rows      SrcB and Dest on by one MVMUL (8 rows): the second half of a face
//   matmul_next_in1_face  SrcA on by one face; SrcB back to the start of its face; Dest on: the next output face
//   matmul_next_in0_face  SrcA back; SrcB on by two faces; Dest on: the bottom output faces
//   matmul_second_k_half  SrcA and SrcB to their second K face; Dest back to the tile's start: accumulate the other half
//   matmul_tile_end       the tile's last MVMUL: SrcA and SrcB back to 0, Dest (and its CR, which holds the tile's
//                         first row) on to the run's next output tile, the fidelity phase to 0
//   matmul_phase_end      HiFi only, the last MVMUL of a fidelity phase that is not the tile's last: SrcA and SrcB
//                         back to 0, Dest back to the tile's first row, the fidelity phase on by one
// With HiFi, the 16 MVMULs run once per fidelity phase. The tile's last MVMUL also releases the streamed Src bank;
// after the last tile of a reused one, the reused bank is released.
//
// Transposed in1 (matmul_modifiers::transpose_in1): each in1 tile t is used as its transpose. The unpacker has
// already transposed inside each face (Haloize_mode, unpack.h), so SrcA holds faces f0', f1', f2', f3' (' = face
// transposed) in rows 0, 16, 32, 48. t's transpose is the 2 x 2 face grid f0', f2' / f1', f3': its face 1 is f2' and
// its face 2 is f1'. Math swaps the two by exchanging SrcA's two steps: matmul_next_in1_face moves SrcA a face row
// (32 rows: f0' -> f2', f1' -> f3') instead of a face, matmul_second_k_half a face (CR to row 16: f1') instead of a face
// row. The MVMUL sequence is unchanged.
//
// Ported from: llk_math_matmul.h (matmul_configure_addrmod, with its transpose values: ADDR_MOD_1 srca += 32,
// ADDR_MOD_4 srca += 16 through CR; the no-MOP form of SDPA 1.0's matmul_block_no_mop).
//
// Not HAL: nothing.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/replay.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h"
#include "llk_lib/thread.h"
#include "resolver/plan.h"
#include "resolver/work_counts.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;

// ---- parameters -----------------------------------------------------------------------------------------------------

/// Parameters of matmul on the math thread. configure and execute take the same struct.
struct matmul_params
{
    std::uint32_t face_r_dim;    ///< rows per face, both operands
    std::uint32_t num_faces;     ///< faces per tile, both operands
    bool transpose;              ///< in1 tiles transposed: the unpacker transposed inside each face, math reads faces 1 and 2 swapped
    bool reuse_in0;              ///< in0 (SrcB) is held while in1 streams; from the Resolver's unpack walk
    std::uint32_t tile_rows;     ///< output tiles along y
    std::uint32_t tile_cols;     ///< output tiles along x
    std::uint32_t k_tiles;       ///< K tiles
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi, 2..4 HiFi2..4 (= fidelity phases)
    bool dest_fp32;              ///< 32-bit Dest mode
    DataFormat src_a_format;     ///< the format in SrcA (in1)
    DataFormat src_b_format;     ///< the format in SrcB (in0)
};

/// The Plan of a matmul as math parameters.
constexpr matmul_params matmul_params_from(const Plan& plan)
{
    return {
        .face_r_dim    = plan.work.face_r_dim,
        .num_faces     = plan.work.num_faces,
        .transpose     = plan.math.transpose_faces,
        .reuse_in0     = plan.walk.held == Where::SrcB,
        .tile_rows     = plan.work.tile_rows,
        .tile_cols     = plan.work.tile_cols,
        .k_tiles       = plan.work.k_tiles,
        .math_fidelity = plan.math.math_fidelity,
        .dest_fp32     = plan.math.dest_fp32,
        .src_a_format  = plan.in1.writes,
        .src_b_format  = plan.in0.writes,
    };
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

// The address modifier slots the MVMULs name. The Dest walk: Dest's CR register holds the tile's first row.
// matmul_tile_end ends the tile; matmul_phase_end ends a fidelity phase that is not the tile's last (HiFi only).
inline constexpr std::uint8_t matmul_next_rows     = 0; ///< SrcB and Dest on by one MVMUL
inline constexpr std::uint8_t matmul_next_in1_face = 1; ///< SrcA on by one face, SrcB back to its CR, Dest on
inline constexpr std::uint8_t matmul_next_in0_face = 2; ///< SrcA back to its CR, SrcB one face row down, Dest on
inline constexpr std::uint8_t matmul_tile_end      = 3; ///< SrcA and SrcB to 0, Dest and its CR on to the run's next output tile, the fidelity phase to 0
inline constexpr std::uint8_t matmul_second_k_half = 4; ///< SrcA and SrcB to their second K face, Dest back to its CR
inline constexpr std::uint8_t matmul_phase_end     = 5; ///< SrcA and SrcB to 0, Dest back to its CR, the fidelity phase on (HiFi)

/// Fidelity phases one tile's MVMULs run: the kernel's fidelity (LoFi: one).
constexpr std::uint32_t matmul_phases(const matmul_params& params)
{
    return params.math_fidelity != 0 ? params.math_fidelity : 1;
}

/// The replay slots of one tile: 16 MVMULs (4 output faces x 2 MVMULs of 8 rows x 2 K halves).
inline constexpr hal::replay::BufferRange matmul_tile_replay {matmul_replay_start, 16};

/// The tile's MVMULs but the last: the last fidelity phase plays these, then the MVMUL that ends the tile.
inline constexpr hal::replay::BufferRange matmul_tile_replay_but_last {matmul_replay_start, matmul_tile_replay.count - 1};

/// Dest rows from one output tile of a run to the next: output tiles are row-major, a run streams along a row when
/// in0 is held and down a column when in1 is.
constexpr std::uint32_t matmul_walk_rows(const matmul_params& params)
{
    return (params.reuse_in0 ? 1 : params.tile_cols) * dest_rows_per_tile(params);
}

/// One MVMUL, then the counters move by address modifier slot AddrMod; Release optionally hands a Src bank back.
template <std::uint8_t AddrMod, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void matmul_mvmul()
{
    hal::fpu::run<hal::fpu::MatrixMultiply {.address_modifier = AddrMod, .release = Release}>();
}

// ---- configure ------------------------------------------------------------------------------------------------------

/// Programs the math thread for P: Dest mode and ALU formats (math_common.h), the address modifiers, the replay of
/// one tile.
template <matmul_params P>
inline __attribute__((always_inline)) void matmul_configure()
{
    using namespace hal::counters::rwc;
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the address modifiers below assume four 16-row faces");
    constexpr bool hifi = P.math_fidelity != 0;

    configure_math_formats<P.dest_fp32, P.src_a_format, P.src_b_format>();

    // Address modifiers (counter_step and how the counters move: math_common.h; a step with .cr set moves the
    // counter's CR register and puts the counter on it). SrcA and SrcB counters are 6 bits
    // (tt-isa-documentation, RWCs.md), so they wrap every 64 rows: one tile's 4 faces of 16 rows. Faces sit in Src
    // row-major: face f starts at row f * face_rows.
    // in1 (SrcA) is K x N: its faces 0, 1 are the first K half, 2, 3 the second (transposed: 0, 2 and 1, 3).
    // in0 (SrcB) is M x K: its faces 0, 2 are the first K half, 1, 3 the second.
    constexpr std::uint32_t face_rows          = P.face_r_dim;
    constexpr std::uint32_t one_face           = face_rows;                      // SrcA: face 0 -> 1, the next output face column
    constexpr std::uint32_t one_face_row       = faces_per_face_row * face_rows; // SrcB: face 0 -> 2, the bottom output faces; SrcA: CR 0 -> face 2
    constexpr std::uint32_t one_face_row_and_1 = one_face_row + face_rows;       // SrcB: CR at face 2 (row 32) + 48 = row 80, wraps to 16: face 1
    // SrcA's two face steps; a transposed in1 swaps them (faces 1 and 2 trade places).
    constexpr counter_step in1_next_face {.incr = P.transpose ? one_face_row : one_face};            // to the next output face column's in1 face
    constexpr counter_step in1_second_half {.incr = P.transpose ? one_face : one_face_row, .cr = 1}; // CR to in1's second K half
    constexpr counter_step in0_next_face_row {.incr = one_face_row, .cr = 1};
    constexpr counter_step in0_second_half {.incr = one_face_row_and_1, .cr = 1};
    constexpr counter_step down_one_mvmul {.incr = rows_per_fpu_op};
    constexpr counter_step dest_next_tile {.incr = matmul_walk_rows(P), .cr = 1};
    static_assert(matmul_walk_rows(P) <= max_dest_counter_step, "the Dest step to the run's next output tile does not fit an address modifier");
    constexpr std::uint32_t next_fidelity_phase = hifi ? 1u : 0u;
    // matmul_next_rows: SrcB and Dest on by one MVMUL, to the second half of the faces
    set_addr_mod<matmul_next_rows, counter_stay, down_one_mvmul, down_one_mvmul>();
    // matmul_next_in1_face: SrcA to in1's next face; SrcB back to the start of its face; Dest on to the next output face
    set_addr_mod<matmul_next_in1_face, in1_next_face, counter_to_cr, down_one_mvmul>();
    // matmul_next_in0_face: SrcA back to its CR face; SrcB (and its CR) one face row down; Dest on to the bottom output faces
    set_addr_mod<matmul_next_in0_face, counter_to_cr, in0_next_face_row, down_one_mvmul>();
    // matmul_second_k_half: SrcA to in1 face 2 (transposed: Src face 1), SrcB to in0 face 1 (both through CR); Dest back to the tile's first row
    set_addr_mod<matmul_second_k_half, in1_second_half, in0_second_half, counter_to_cr>();
    // matmul_phase_end: SrcA and SrcB to 0, Dest back to the tile's first row (its CR), and the next fidelity phase
    set_addr_mod<matmul_phase_end, counter_clear, counter_clear, counter_to_cr, next_fidelity_phase>();
    // matmul_tile_end: SrcA and SrcB to 0, Dest's CR (and Dest) on to the run's next output tile, the fidelity phase to 0
    set_addr_mod<matmul_tile_end, counter_clear, counter_clear, dest_next_tile, 0 /*FidelityIncr*/, 1 /*FidelityClear*/>();
    constexpr std::uint8_t last_recorded = hifi ? matmul_phase_end : matmul_tile_end; // HiFi: execute issues the tile's last MVMUL itself

    // One tile: 16 MVMULs. With LoFi the last one also releases the streamed Src bank; with HiFi that waits for the
    // last phase's last MVMUL, issued by execute.
    constexpr auto release_streamed = hifi ? hal::fpu::SourceRelease::None : (P.reuse_in0 ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::SrcB);
    constexpr std::uint32_t k_halves_per_tile = faces_per_face_row; // in0's two K faces (in1's two K face rows)
    constexpr std::uint32_t mvmuls_per_tile   = P.num_faces * (P.face_r_dim / rows_per_fpu_op) * k_halves_per_tile;
    static_assert(mvmuls_per_tile == matmul_tile_replay.count, "the replay below records 16 MVMULs: 4 output faces x 2 MVMULs of 8 rows x 2 K halves");
    hal::replay::record<matmul_tile_replay>(
        []
        {
            // First K half: output faces 0, 1 (top), then 2, 3 (bottom), two MVMULs (16 rows) each.
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_next_in1_face>();
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_next_in0_face>();
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_next_in1_face>();
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_second_k_half>();
            // Second K half, accumulated onto the same four faces.
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_next_in1_face>();
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_next_in0_face>();
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<matmul_next_in1_face>();
            matmul_mvmul<matmul_next_rows>();
            matmul_mvmul<last_recorded, release_streamed>();
        });

    reset_math_counters();
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// One K step into the tile_rows x tile_cols output tiles starting at dest_row (an absolute Dest row: the Dest tensor's
/// address). Walks the tiles in the order the unpack walk delivers them: for each held (reused) tile, every streamed
/// one. streamed: the streamed tiles of a run, P's (tile_cols with in0 held, else tile_rows) or a run-time count of at
/// most that.
template <matmul_params P, class Streamed>
inline __attribute__((always_inline)) void matmul_k_step(std::uint32_t dest_row, const Streamed& streamed)
{
    using namespace hal::counters::rwc;
    constexpr std::uint32_t reused_tiles = P.reuse_in0 ? P.tile_rows : P.tile_cols;
    // The reused bank back to the unpacker, after its last streamed tile: with every counter to 0, or with Dest left
    // where the walk below has it.
    constexpr MathCounter release_reused {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = P.reuse_in0 ? Src::B : Src::A, .clear_fidelity = true};
    constexpr MathCounter release_reused_keep_dest {.src_a = Set(0), .src_b = Set(0), .flip = P.reuse_in0 ? Src::B : Src::A, .clear_fidelity = true};
    constexpr auto release_streamed = P.reuse_in0 ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::SrcB;

    // The Dest row counter walks the output tiles: its CR register is the tile's first row, counted from the math Dest
    // offset, and the tile's last MVMUL moves it on (matmul_tile_end). With in0 held the runs follow each other in Dest
    // (row-major), so the offset is written once and the counter walks the whole K step; with in1 held a run goes down
    // a column, and each run starts from the offset of its column's first tile.
    if constexpr (P.reuse_in0)
    {
        set_math_dest_tile<P>(dest_row, 0);
    }
    for (std::uint32_t reused_tile = 0; reused_tile < reused_tiles; reused_tile++)
    {
        if constexpr (!P.reuse_in0)
        {
            set_math_dest_tile<P>(dest_row, reused_tile);
        }
        for (std::uint32_t streamed_tile = 0; streamed_tile < wrapper::value(streamed); streamed_tile++)
        {
            if constexpr (P.math_fidelity != 0)
            {
                // Every phase but the last from the replay; the last one's final MVMUL ends the tile instead of the
                // phase and hands the streamed bank back.
#pragma GCC unroll 4
                for (std::uint32_t phase = 0; phase + 1 < matmul_phases(P); phase++)
                {
                    hal::replay::run<matmul_tile_replay>();
                }
                hal::replay::run<matmul_tile_replay_but_last>();
                matmul_mvmul<matmul_tile_end, release_streamed>();
            }
            else
            {
                hal::replay::run<matmul_tile_replay>(); // its last MVMUL is the tile's last
            }
        }
        // The reused tile is done: release its bank. After the K step's last run Dest and its CR go back to 0.
        if (P.reuse_in0 && reused_tile != reused_tiles - 1)
        {
            set<release_reused_keep_dest>();
        }
        else
        {
            set<release_reused>();
        }
    }
}

/// Every K step, each accumulating into the same output tiles in Dest from dest_row (absolute). k_steps is their
/// count: a ct (P.k_tiles) or, for a K known only at run time, an rt of at most that (resolver/work_counts.h).
/// streamed: the streamed tiles of each run, the same way (a run-time number of output tile columns).
template <matmul_params P, class Steps, class Streamed, ctrt<Steps, std::uint32_t> = 0, ctrt<Streamed, std::uint32_t> = 0>
inline __attribute__((always_inline)) void matmul_execute(std::uint32_t dest_row, const Steps& k_steps, const Streamed& streamed)
{
    if constexpr (wrapper::is_rt_v<Steps>)
    {
        LLK_ASSERT(wrapper::value(k_steps) >= 1 && wrapper::value(k_steps) <= P.k_tiles, "a run-time K is outside 1..its bound");
    }
    if constexpr (wrapper::is_rt_v<Streamed>)
    {
        LLK_ASSERT(
            wrapper::value(streamed) >= 1 && wrapper::value(streamed) <= (P.reuse_in0 ? P.tile_cols : P.tile_rows),
            "a run-time number of streamed tiles is outside 1..its bound");
    }
    for (std::uint32_t k = 0; k < wrapper::value(k_steps); k++)
    {
        matmul_k_step<P>(dest_row, streamed);
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The math thread's part of a Compute API matmul: configure, then every K step into Dest from dest_row. counts is
/// the run-time side of the Plan's Work (an empty object when every count is compile time).
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void matmul(const WorkCounts<Pl.work>& counts, std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    // matmul_execute takes the runs of a K step as P's number. The Resolver gives a run-time N only with one in0 tile
    // row, which makes in0 the held operand and its one tile the run.
    static_assert(
        !wrapper::is_rt_v<decltype(plan_count<Pl.walk.runs_of, Pl.walk.runs>(counts))>, "matmul: the held tiles of a K step are a compile-time count");
    constexpr matmul_params params = matmul_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        matmul_configure<params>();
        // The K steps, and the streamed tiles of a run: the unpack walk's steps and tiles_per_run.
        matmul_execute<params>(dest_row, plan_count<Pl.walk.steps_of, Pl.walk.steps>(counts), plan_count<Pl.walk.tiles_of, Pl.walk.tiles_per_run>(counts));
    }
}

} // namespace llk2::llk_lib
