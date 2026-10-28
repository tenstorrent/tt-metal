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
//   next_rows      SrcB and Dest on by one MVMUL (8 rows): the second half of a face
//   next_in1_face  SrcA on by one face; SrcB back to the start of its face; Dest on: the next output face
//   next_in0_face  SrcA back; SrcB on by two faces; Dest on: the bottom output faces
//   second_k_half  SrcA and SrcB to their second K face; Dest back to the tile's start: accumulate the other half
//   tile_done      everything back to 0, and the fidelity phase on by one (HiFi)
// With HiFi, the MOP repeats the 16 MVMULs once per fidelity phase. After the tile, the streamed Src bank is
// released; after the last tile of a reused one, the reused bank is.
//
// Not HAL yet: the Src zero-flag rule (requires_disabled_src_zero_flag, cmath_common.h).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
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
    bool reuse_x;                ///< x (SrcB) is held while w streams; from the Resolver's unpack walk
    std::uint32_t rt_dim;        ///< output tiles along y
    std::uint32_t ct_dim;        ///< output tiles along x
    std::uint32_t kt_dim;        ///< K tiles
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi, 2..4 HiFi2..4 (= fidelity phases)
    bool dest_fp32;              ///< 32-bit Dest mode
    DataFormat math_format;      ///< the Src format the FPU reads
};

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

inline constexpr std::uint32_t mvmul_rows = 8; ///< SrcB and Dest rows one MVMUL covers (and SrcA is one 16-row face)
inline constexpr std::uint32_t faces_per_face_row = 2; ///< a 32x32 tile is 2 x 2 faces, stored face 0, 1 (top), 2, 3 (bottom)
inline constexpr std::uint32_t replay_tile_start  = 16; ///< the math thread's replay slots for one tile: [16, 32)

/// Dest rows one output tile takes.
constexpr std::uint32_t dest_rows_per_tile(const matmul_params& p)
{
    return p.num_faces * p.face_r_dim;
}

/// Row counter behavior of one address modifier: {increment, clear, carriage return} for SrcA, SrcB and Dest.
struct counter_step
{
    std::uint32_t incr, clr, cr;
};

/// Programs address modifier Slot. Fields this does not name are written 0, as addr_mod_t does.
template <cfg::Sec Slot, counter_step A, counter_step B, counter_step D, std::uint32_t FidelityIncr = 0>
inline __attribute__((always_inline)) void set_addr_mod()
{
    using cfg::AddrMod;
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
        cfg::set<AddrMod[cfg::Fidelity].Incr, Slot, FidelityIncr>(),
        cfg::set<AddrMod[cfg::Fidelity].Clear, Slot, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, AddrMod[cfg::Bias].Incr, Slot, 0>();
}

// The address modifier slots the replay uses (the ADDR_MOD_n numbers of the original LLK).
inline constexpr std::uint8_t next_rows     = 0;
inline constexpr std::uint8_t next_in1_face = 1;
inline constexpr std::uint8_t next_in0_face = 2;
inline constexpr std::uint8_t second_k_half = 4;
inline constexpr std::uint8_t tile_done     = 5;

template <std::uint8_t AddrMod, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void mvmul()
{
    hal::fpu::run<hal::fpu::MatrixMultiply {.address_modifier = AddrMod, .release = Release}>();
}

/// Programs the math thread for P: Dest mode, ALU formats, address modifiers, the replay and the MOP.
template <matmul_params P>
inline __attribute__((always_inline)) void matmul_configure()
{
    using namespace ckernel;
    using namespace hal::counters::rwc;
    static_assert(P.num_faces == 4 && P.face_r_dim == FACE_R_DIM, "32x32 tiles only, for now: the address modifiers below assume four 16-row faces");
    static_assert(!P.transpose, "transposed matmul is not ported yet");
    constexpr bool hifi    = P.math_fidelity != 0;
    constexpr bool reuse_b = P.reuse_x;

    // Dest mode and ALU format baseline (the math side of hw_configure). Wait for math and SFPU to drain first.
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
    constexpr std::uint32_t int8_math = is_int8_or_int32_format(static_cast<std::uint32_t>(P.math_format)) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, int8_math>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());

    // Address modifiers. Row counters (RWCs) follow the ISA's ApplyAddrMod: a plain step adds incr to the counter; a
    // carriage-return step adds incr to the counter's CR register and copies it to the counter; clear zeroes both.
    // SrcA and SrcB counters are 6 bits (tt-isa-documentation, RWCs.md), so they wrap every 64 rows: one tile's
    // 4 faces of 16 rows. Faces sit in Src row-major: face f starts at row f * face_rows.
    // in1 (SrcA) is K x N: its faces 0, 1 are the first K half, 2, 3 the second.
    // in0 (SrcB) is M x K: its faces 0, 2 are the first K half, 1, 3 the second.
    constexpr std::uint32_t face_rows          = P.face_r_dim;
    constexpr std::uint32_t one_face           = face_rows;                      // SrcA: face 0 -> 1, the next output face column
    constexpr std::uint32_t one_face_row       = faces_per_face_row * face_rows; // SrcB: face 0 -> 2, the bottom output faces; SrcA: CR 0 -> face 2
    constexpr std::uint32_t one_face_row_and_1 = one_face_row + face_rows;       // SrcB: CR at face 2 (row 32) + 48 = row 80, wraps to 16: face 1
    constexpr counter_step stay {0, 0, 0};          // counter unchanged
    constexpr counter_step back_to_cr {0, 0, 1};    // counter back to its CR row
    constexpr counter_step clear {0, 1, 1};         // counter and CR to 0
    constexpr counter_step down_one_mvmul {mvmul_rows, 0, 0};
    // next_rows: SrcB and Dest on by one MVMUL, to the second half of the faces
    set_addr_mod<cfg::Sec::S0, stay, down_one_mvmul, down_one_mvmul>();
    // next_in1_face: SrcA to in1's next face; SrcB back to the start of its face; Dest on to the next output face
    set_addr_mod<cfg::Sec::S1, counter_step {one_face, 0, 0}, back_to_cr, down_one_mvmul>();
    // next_in0_face: SrcA back to its CR face; SrcB (and its CR) one face row down; Dest on to the bottom output faces
    set_addr_mod<cfg::Sec::S2, back_to_cr, counter_step {one_face_row, 0, 1}, down_one_mvmul>();
    // second_k_half: SrcA to in1 face 2, SrcB to in0 face 1 (both through CR); Dest back to the tile's first row
    set_addr_mod<cfg::Sec::S4, counter_step {one_face_row, 0, 1}, counter_step {one_face_row_and_1, 0, 1}, back_to_cr>();
    // tile_done: every counter to 0, and the next fidelity phase (HiFi only)
    set_addr_mod<cfg::Sec::S5, clear, clear, clear, hifi ? 1u : 0u>();

    // One tile: 16 MVMULs. With LoFi the last one also releases the streamed Src bank; with HiFi that waits for the
    // MOP's end op, after the last fidelity phase.
    constexpr auto release_streamed = hifi ? hal::fpu::SourceRelease::None : (reuse_b ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::SrcB);
    constexpr std::uint32_t mvmuls_per_tile = 16; // 4 output faces x 2 MVMULs (8 rows each) x 2 K halves
    constexpr hal::replay::BufferRange tile_replay {replay_tile_start, mvmuls_per_tile};
    hal::replay::record<tile_replay>(
        []
        {
            // First K half: output faces 0, 1 (top), then 2, 3 (bottom), two MVMULs (16 rows) each.
            mvmul<next_rows>();
            mvmul<next_in1_face>();
            mvmul<next_rows>();
            mvmul<next_in0_face>();
            mvmul<next_rows>();
            mvmul<next_in1_face>();
            mvmul<next_rows>();
            mvmul<second_k_half>();
            // Second K half, accumulated onto the same four faces.
            mvmul<next_rows>();
            mvmul<next_in1_face>();
            mvmul<next_rows>();
            mvmul<next_in0_face>();
            mvmul<next_rows>();
            mvmul<next_in1_face>();
            mvmul<next_rows>();
            mvmul<tile_done, release_streamed>();
        });

    // MOP template 1: the tile replay once per fidelity phase (LoFi: once). HiFi releases the streamed bank and
    // resets the counters after the last phase.
    constexpr std::uint32_t tile_op = hal::replay::get_operation<tile_replay>();
    if constexpr (hifi)
    {
        constexpr std::uint32_t release_streamed_op =
            MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = reuse_b ? Src::A : Src::B, .clear_fidelity = true}.operation();
        hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
            .outer_loop = {.count = 1, .end_op = release_streamed_op}, .inner_loop = {.count = P.math_fidelity, .body_op = tile_op}}>();
    }
    else
    {
        hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {.outer_loop = {.count = 1}, .inner_loop = {.count = 1, .body_op = tile_op}}>();
    }
    set<MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .clear_fidelity = true}>();

    // Src zero-flag baseline for this format; the ckernel caches stay in step for the LLK asserts.
    const std::uint32_t math_format = static_cast<std::uint32_t>(P.math_format);
    math::src_zero_flag_srca_fmt    = math_format;
    math::src_zero_flag_srcb_fmt    = math_format;
    const std::uint32_t zero_flag   = requires_disabled_src_zero_flag(math_format, math_format) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(zero_flag);
    math::src_zero_flag_hw = zero_flag;
}

/// One K step into the rt_dim x ct_dim output tiles starting at dest_row (an absolute Dest row: the Dest tensor's address).
/// Walks the tiles in the order the unpack walk delivers them: for each held (reused) tile, every streamed one.
template <matmul_params P>
inline __attribute__((always_inline)) void matmul_execute(std::uint32_t dest_row)
{
    using namespace ckernel;
    using namespace hal::counters::rwc;
    constexpr bool reuse_b                 = P.reuse_x;
    constexpr std::uint32_t reused_tiles   = reuse_b ? P.rt_dim : P.ct_dim;
    constexpr std::uint32_t streamed_tiles = reuse_b ? P.ct_dim : P.rt_dim;
    constexpr MathCounter release_reused {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = reuse_b ? Src::B : Src::A, .clear_fidelity = true};

    for (std::uint32_t reused_tile = 0; reused_tile < reused_tiles; reused_tile++)
    {
        for (std::uint32_t streamed_tile = 0; streamed_tile < streamed_tiles; streamed_tile++)
        {
            // Output tiles are row-major: (row, column) = (reused, streamed) when x is reused, else swapped.
            const std::uint32_t output_tile = reuse_b ? P.ct_dim * reused_tile + streamed_tile : reused_tile + streamed_tile * P.ct_dim;
            cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row + output_tile * dest_rows_per_tile(P));
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
