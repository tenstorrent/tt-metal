// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, matmul on Blackhole. Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/). Math only:
// the unpack side of matmul is in unpack.h.
//
//   matmul_params                one struct; matmul_configure and matmul_execute take the same one
//   matmul_params_from(Plan)     fills it from the Resolver's Plan
//   matmul_configure<P>()        Dest mode and ALU formats, the address modifiers, the replay of one tile's MVMULs
//                                and the MOP that repeats it once per fidelity phase
//   matmul_execute<P>(row)       one K step: every output tile gets in0's tile k (SrcB) times in1's tile k (SrcA)
//                                added to it in Dest
//   matmul<On, Plan>(row)        the per-thread call the Compute API makes: configures and runs all K steps, only
//                                when this build is for thread On (the math thread)
//
// How one tile is computed. An MVMUL multiplies 8 rows of SrcB (8 x 16 datums of an in0 face) by a 16 x 16 SrcA
// face (in1) and adds the 8 x 16 product into 8 rows of Dest. A 32x32 output tile has 4 faces of 16 rows, and
// each needs two products along K (in0's two K faces times in1's), so a tile is 16 MVMULs. They run from a replay
// buffer; between them, the address modifiers (ADDR_MODs) move the SrcA, SrcB and Dest row counters:
//   mm_next_rows      SrcB and Dest on by one MVMUL (8 rows): the second half of a face
//   mm_next_in1_face  SrcA on by one face; SrcB back to the start of its face; Dest on: the next output face
//   mm_next_in0_face  SrcA back; SrcB on by two faces; Dest on: the bottom output faces
//   mm_second_k_half  SrcA and SrcB to their second K face; Dest back to the tile's start: accumulate the other half
//   mm_tile_done      everything back to 0, and the fidelity phase on by one (HiFi)
// With HiFi, the MOP repeats the 16 MVMULs once per fidelity phase. After the tile, the streamed Src bank is
// released; after the last tile of a reused one, the reused bank is.
//
// Shared math configure (and its non-HAL parts): math_common.h.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h"
#include "llk_lib/thread.h"
#include "resolver/plan.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// Parameters of matmul on the math thread. configure and execute take the same struct.
struct matmul_params
{
    std::uint32_t face_r_dim;    ///< rows per face, both operands
    std::uint32_t num_faces;     ///< faces per tile, both operands
    bool transpose;              ///< in1's faces transposed (not ported yet)
    bool reuse_in0;              ///< in0 (SrcB) is held while in1 streams; from the Resolver's unpack walk
    std::uint32_t rt_dim;        ///< output tiles along y
    std::uint32_t ct_dim;        ///< output tiles along x
    std::uint32_t kt_dim;        ///< K tiles
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi, 2..4 HiFi2..4 (= fidelity phases)
    bool dest_fp32;              ///< 32-bit Dest mode
    DataFormat math_format;      ///< the Src format the FPU reads
};

/// The Plan of a matmul as math parameters.
constexpr matmul_params matmul_params_from(const Plan& p)
{
    return {
        p.work.face_r_dim,
        p.work.num_faces,
        p.math.transpose_faces,
        p.walk.held == Where::SrcB,
        p.work.rt_dim,
        p.work.ct_dim,
        p.work.kt_dim,
        p.math.math_fidelity,
        p.math.dest_fp32,
        p.math.format,
    };
}

// The address modifier slots the replay uses (the ADDR_MOD_n numbers of the original LLK). Slot 3 is unused.
inline constexpr std::uint8_t mm_next_rows     = 0; ///< SrcB and Dest on by one MVMUL
inline constexpr std::uint8_t mm_next_in1_face = 1; ///< SrcA on by one face, SrcB back to its CR, Dest on
inline constexpr std::uint8_t mm_next_in0_face = 2; ///< SrcA back to its CR, SrcB one face row down, Dest on
inline constexpr std::uint8_t mm_second_k_half = 4; ///< SrcA and SrcB to their second K face, Dest back to its CR
inline constexpr std::uint8_t mm_tile_done     = 5; ///< every counter to 0, the fidelity phase on (HiFi)

/// One MVMUL, then the counters move by address modifier slot AddrMod; Release optionally hands a Src bank back.
template <std::uint8_t AddrMod, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void mvmul()
{
    hal::fpu::run<hal::fpu::MatrixMultiply {.address_modifier = AddrMod, .release = Release}>();
}

/// Programs the math thread for P: Dest mode, ALU formats, address modifiers, the replay and the MOP.
template <matmul_params P>
inline __attribute__((always_inline)) void matmul_configure()
{
    using namespace hal::counters::rwc;
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the address modifiers below assume four 16-row faces");
    static_assert(!P.transpose, "transposed matmul is not ported yet");
    constexpr bool hifi = P.math_fidelity != 0;

    configure_math_formats<P.dest_fp32, P.math_format, P.math_format>();

    // Address modifiers (counter_step and how the counters move: math_common.h). SrcA and SrcB counters are 6 bits
    // (tt-isa-documentation, RWCs.md), so they wrap every 64 rows: one tile's 4 faces of 16 rows. Faces sit in Src
    // row-major: face f starts at row f * face_rows.
    // in1 (SrcA) is K x N: its faces 0, 1 are the first K half, 2, 3 the second.
    // in0 (SrcB) is M x K: its faces 0, 2 are the first K half, 1, 3 the second.
    constexpr std::uint32_t face_rows          = P.face_r_dim;
    constexpr std::uint32_t one_face           = face_rows;                      // SrcA: face 0 -> 1, the next output face column
    constexpr std::uint32_t one_face_row       = faces_per_face_row * face_rows; // SrcB: face 0 -> 2, the bottom output faces; SrcA: CR 0 -> face 2
    constexpr std::uint32_t one_face_row_and_1 = one_face_row + face_rows;       // SrcB: CR at face 2 (row 32) + 48 = row 80, wraps to 16: face 1
    constexpr counter_step back_to_cr {0, 0, 1};                                 // counter back to its CR row
    constexpr counter_step down_one_mvmul {rows_per_fpu_op, 0, 0};
    // mm_next_rows: SrcB and Dest on by one MVMUL, to the second half of the faces
    set_addr_mod<static_cast<cfg::Sec>(mm_next_rows), counter_stay, down_one_mvmul, down_one_mvmul>();
    // mm_next_in1_face: SrcA to in1's next face; SrcB back to the start of its face; Dest on to the next output face
    set_addr_mod<static_cast<cfg::Sec>(mm_next_in1_face), counter_step {one_face, 0, 0}, back_to_cr, down_one_mvmul>();
    // mm_next_in0_face: SrcA back to its CR face; SrcB (and its CR) one face row down; Dest on to the bottom output faces
    set_addr_mod<static_cast<cfg::Sec>(mm_next_in0_face), back_to_cr, counter_step {one_face_row, 0, 1}, down_one_mvmul>();
    // mm_second_k_half: SrcA to in1 face 2, SrcB to in0 face 1 (both through CR); Dest back to the tile's first row
    set_addr_mod<static_cast<cfg::Sec>(mm_second_k_half), counter_step {one_face_row, 0, 1}, counter_step {one_face_row_and_1, 0, 1}, back_to_cr>();
    // mm_tile_done: every counter to 0, and the next fidelity phase (HiFi only)
    constexpr std::uint32_t next_fidelity_phase = hifi ? 1u : 0u;
    set_addr_mod<static_cast<cfg::Sec>(mm_tile_done), counter_clear, counter_clear, counter_clear, next_fidelity_phase>();

    // One tile: 16 MVMULs. With LoFi the last one also releases the streamed Src bank; with HiFi that waits for the
    // MOP's end op, after the last fidelity phase.
    constexpr auto release_streamed = hifi ? hal::fpu::SourceRelease::None : (P.reuse_in0 ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::SrcB);
    constexpr std::uint32_t k_halves_per_tile = faces_per_face_row; // in0's two K faces (in1's two K face rows)
    constexpr std::uint32_t mvmuls_per_tile   = P.num_faces * (P.face_r_dim / rows_per_fpu_op) * k_halves_per_tile;
    static_assert(mvmuls_per_tile == 16, "the replay below records 16 MVMULs: 4 output faces x 2 MVMULs of 8 rows x 2 K halves");
    constexpr hal::replay::BufferRange tile_replay {math_replay_tile_start, mvmuls_per_tile};
    hal::replay::record<tile_replay>(
        []
        {
            // First K half: output faces 0, 1 (top), then 2, 3 (bottom), two MVMULs (16 rows) each.
            mvmul<mm_next_rows>();
            mvmul<mm_next_in1_face>();
            mvmul<mm_next_rows>();
            mvmul<mm_next_in0_face>();
            mvmul<mm_next_rows>();
            mvmul<mm_next_in1_face>();
            mvmul<mm_next_rows>();
            mvmul<mm_second_k_half>();
            // Second K half, accumulated onto the same four faces.
            mvmul<mm_next_rows>();
            mvmul<mm_next_in1_face>();
            mvmul<mm_next_rows>();
            mvmul<mm_next_in0_face>();
            mvmul<mm_next_rows>();
            mvmul<mm_next_in1_face>();
            mvmul<mm_next_rows>();
            mvmul<mm_tile_done, release_streamed>();
        });

    // MOP template 1: the tile replay once per fidelity phase (LoFi: once). HiFi releases the streamed bank and
    // resets the counters after the last phase.
    constexpr std::uint32_t tile_op = hal::replay::get_operation<tile_replay>();
    if constexpr (hifi)
    {
        constexpr std::uint32_t release_streamed_op =
            MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = P.reuse_in0 ? Src::A : Src::B, .clear_fidelity = true}.operation();
        hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
            .outer_loop = {.count = 1 /* one pass */, .end_op = release_streamed_op},
            .inner_loop = {.count = P.math_fidelity /* phases */, .body_op = tile_op}}>();
    }
    else
    {
        // One pass of the outer loop, one of the inner: the tile replay once.
        hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {.outer_loop = {.count = 1}, .inner_loop = {.count = 1, .body_op = tile_op}}>();
    }
    reset_math_counters();
}

/// One K step into the rt_dim x ct_dim output tiles starting at dest_row (an absolute Dest row: the Dest tensor's address).
/// Walks the tiles in the order the unpack walk delivers them: for each held (reused) tile, every streamed one.
template <matmul_params P>
inline __attribute__((always_inline)) void matmul_execute(std::uint32_t dest_row)
{
    using namespace hal::counters::rwc;
    constexpr std::uint32_t reused_tiles   = P.reuse_in0 ? P.rt_dim : P.ct_dim;
    constexpr std::uint32_t streamed_tiles = P.reuse_in0 ? P.ct_dim : P.rt_dim;
    constexpr MathCounter release_reused {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = P.reuse_in0 ? Src::B : Src::A, .clear_fidelity = true};

    for (std::uint32_t reused_tile = 0; reused_tile < reused_tiles; reused_tile++)
    {
        for (std::uint32_t streamed_tile = 0; streamed_tile < streamed_tiles; streamed_tile++)
        {
            // Output tiles are row-major: (row, column) = (reused, streamed) when in0 is reused, else swapped.
            const std::uint32_t output_tile = P.reuse_in0 ? P.ct_dim * reused_tile + streamed_tile : reused_tile + streamed_tile * P.ct_dim;
            set_math_dest_tile<P>(dest_row, output_tile);
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
            if (streamed_tile == streamed_tiles - 1)
            {
                set<release_reused>(); // the reused tile is done: release its bank
            }
        }
    }
}

/// The math thread's part of a Compute API matmul: configure, then all K steps into Dest from dest_row.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void matmul(std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    constexpr matmul_params p = matmul_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        matmul_configure<p>();
        for (std::uint32_t k = 0; k < p.kt_dim; k++)
        {
            matmul_execute<p>(dest_row);
        }
    }
}

} // namespace llk2::llk_lib
