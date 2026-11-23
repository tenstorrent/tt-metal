// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, eltwise binary (add, sub, mul) on Blackhole. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/). The FPU's part of a Compute API add, sub or mul; the unpackers' part is unpack.h.
// The shape of an op: llk_lib/thread.h.
//
//   eltwise_binary_params                 one struct; configure and execute take the same one
//   eltwise_binary_params_from(Plan)      fills it from the Resolver's Plan
//   eltwise_binary_configure<P>()         Dest mode and ALU formats, the address modifiers, the replay of one tile's
//                                         eight FPU instructions (no MOP: execute plays the replay once per fidelity
//                                         phase, math_common.h)
//   eltwise_binary_execute<P>(dest_row, tiles)
//                                         every output tile: Dest = SrcA <function> SrcB, from Dest row dest_row.
//                                         tiles is their count: a ct (P's tile_rows * tile_cols) or, for a size known
//                                         only at run time, an rt of at most that (resolver/work_counts.h)
//   eltwise_binary<On, Plan>(counts, dest_row)
//                                         the per-thread call the Compute API makes: configures and executes, only
//                                         when this build is for thread On (the math thread). counts (WorkCounts) is the
//                                         run-time side of the Plan's Work
//
// How one tile is computed. The unpackers hand math one whole tile per Src bank: SrcA rows 0..63 hold in0's faces
// 0..3. An ELWADD (ELWSUB, ELWMUL) takes 8 rows of SrcA and 8 rows of SrcB and writes 8 rows of Dest, so a 32x32
// tile is 8 instructions. They run from a replay buffer; between them, the address modifiers move the row counters:
//   eltwise_next_rows       the second half of a face
//   eltwise_next_face_col   the next face in the face row (face 0 -> 1, 2 -> 3)
//   eltwise_next_face_row   the first face of the next face row (face 1 -> 2)
//   eltwise_pass_end        the last of the 8: SrcA and SrcB back to 0. With one phase Dest steps on like the others,
//                           to the next tile; with HiFi (mul) Dest goes back to the tile's first row and the fidelity
//                           phase on by one, and after the last phase eltwise_tile_end ends the tile instead
// The first three all move SrcA and Dest on by 8 rows; they differ only in SrcB, by the broadcast (the FPU's SrcB
// broadcast, tt-isa-documentation ELWADD.md: Row reads the one SrcB row its counter selects, unaligned, and repeats
// it down all 8 rows; Col reads column 0 of each of the 8 rows and repeats it across; Scalar both):
//   None    SrcB holds in1's whole tile and moves with SrcA.
//   Row     SrcB holds only in1's top face row (rows 0..15 face 0, rows 16..31 face 1). SrcB stays on row 0 through
//           face 0, moves to row 16 (face 1's row 0) for face 1, and back to row 0 and 16 for faces 2 and 3.
//   Col     SrcB holds in1's whole tile; tile row r's column 0 is in face 0 (r < 16) or face 2. SrcB walks face 0's
//           rows for face 0, back (to its CR row, 0) for face 1, on to face 2's rows (CR 32) for face 2, back to 32
//           for face 3.
//   Scalar  SrcB holds in1's face 0; its counter stays on row 0, so every instruction reads datum [0, 0].
// The last instruction of the tile releases both Src banks to the unpackers.
//
// mul. ELWMUL multiplies only the top bits of each mantissa per fidelity phase and always adds its product to Dest
// (ISA: ELWMUL.md). So the tile's 8 instructions run once per phase (HiFi2: 2,
// ..., HiFi4: 4; LoFi: once), the phases summing in Dest: the last instruction of each pass moves the fidelity phase
// on and the counters back to the tile's first row, and after the last pass both Src banks are released and the
// phase cleared. For mul to write Dest = in0 * in1 (as add and sub do) rather than add to it, execute first zeroes each
// output tile's faces (ZEROACC: zeroed rows read as 0 to ELWMUL); the 1.0 LLK instead relies on the packer having
// zeroed Dest. mul with .accumulate skips the zeroing, so the phases add the product to what Dest already holds:
// Dest = Dest + in0 * in1 (SDPA: P V, then the correction factor times the previous output added into it).
//
// Ported from: llk_math_eltwise_binary.h (the standard path: no Dest reuse, no accumulation; its ELWMUL fidelity
// loop).
//
// Not HAL: nothing.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/dst.h"
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

/// Parameters of an eltwise binary op on the math thread. configure and execute take the same struct.
struct eltwise_binary_params
{
    std::uint32_t face_r_dim; ///< rows per face, both operands
    std::uint32_t num_faces;  ///< faces per tile, both operands
    std::uint32_t tile_rows;  ///< output tiles along y
    std::uint32_t tile_cols;  ///< output tiles along x
    BinaryFunction function;  ///< add, sub or mul
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi (one phase), 2..4 HiFi2..4 (= phases); mul only
    Broadcast broadcast;      ///< how SrcB is spread over SrcA
    bool accumulate;          ///< mul only: add the product to Dest (no zeroing first)
    bool dest_fp32;           ///< 32-bit Dest mode
    DataFormat src_a_format;  ///< the format in SrcA
    DataFormat src_b_format;  ///< the format in SrcB
};

/// The operand of a Plan that the unpackers write to `where`.
constexpr const UnpackPlan& eltwise_binary_operand_in(const Plan& plan, Where where)
{
    return plan.in0.to == where ? plan.in0 : plan.in1;
}

/// The Plan of an eltwise binary op as math parameters.
constexpr eltwise_binary_params eltwise_binary_params_from(const Plan& plan)
{
    return {
        .face_r_dim    = plan.work.face_r_dim,
        .num_faces     = plan.work.num_faces,
        .tile_rows     = plan.work.tile_rows,
        .tile_cols     = plan.work.tile_cols,
        .function      = plan.math.binary,
        .math_fidelity = plan.math.math_fidelity,
        .broadcast     = eltwise_binary_operand_in(plan, Where::SrcB).broadcast,
        .accumulate    = plan.math.accumulate,
        .dest_fp32     = plan.math.dest_fp32,
        .src_a_format  = eltwise_binary_operand_in(plan, Where::SrcA).writes,
        .src_b_format  = eltwise_binary_operand_in(plan, Where::SrcB).writes,
    };
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

inline constexpr std::uint32_t eltwise_ops_per_tile = faces_per_full_tile * (max_face_rows / rows_per_fpu_op); ///< 4 faces x 2 instructions of 8 rows

/// The FPU instruction for a BinaryFunction: ELWADD, ELWSUB or ELWMUL.
constexpr hal::fpu::ElementwiseOperation eltwise_fpu_operation(BinaryFunction f)
{
    return f == BinaryFunction::Mul   ? hal::fpu::ElementwiseOperation::Multiply
           : f == BinaryFunction::Sub ? hal::fpu::ElementwiseOperation::Subtract
                                      : hal::fpu::ElementwiseOperation::Add;
}

/// The FPU's SrcB broadcast for an operand's Broadcast.
constexpr hal::fpu::SrcBBroadcast eltwise_fpu_broadcast(Broadcast b)
{
    switch (b)
    {
        case Broadcast::Row:
            return hal::fpu::SrcBBroadcast::Row;
        case Broadcast::Col:
            return hal::fpu::SrcBBroadcast::Column;
        case Broadcast::Scalar:
            return hal::fpu::SrcBBroadcast::Scalar;
        default:
            return hal::fpu::SrcBBroadcast::None;
    }
}

/// The tile's instructions run once per fidelity phase: mul with HiFi (ELWADD and ELWSUB take one phase).
constexpr bool eltwise_hifi(const eltwise_binary_params& params)
{
    return params.function == BinaryFunction::Mul && params.math_fidelity != 0;
}

/// Fidelity phases one tile's instructions run: mul's with HiFi, else one.
constexpr std::uint32_t eltwise_phases(const eltwise_binary_params& params)
{
    return eltwise_hifi(params) ? params.math_fidelity : 1;
}

/// The replay slots of one tile: its 8 instructions.
inline constexpr hal::replay::BufferRange eltwise_tile_replay {eltwise_replay_start, eltwise_ops_per_tile};

/// HiFi: the counters back to 0, the fidelity phase cleared and both Src banks released, after a tile's last phase.
/// (One phase: the tile's last instruction releases them.)
inline constexpr hal::counters::rwc::MathCounter eltwise_release_both {
    .src_a          = hal::counters::rwc::Set(0),
    .src_b          = hal::counters::rwc::Set(0),
    .dest           = hal::counters::rwc::Set(0),
    .flip           = hal::counters::rwc::Src::A | hal::counters::rwc::Src::B,
    .clear_fidelity = true};

// The address modifier slots the instructions name. The Dest walk: the Dest row counter walks from tile to tile.
// With one phase eltwise_pass_end lets Dest step on like every other instruction of the tile (the next tile follows
// in Dest). With HiFi Dest's CR register holds the tile's first row: eltwise_pass_end ends a phase that is not the
// last (Dest back to its CR), and eltwise_tile_end ends the tile.
inline constexpr std::uint8_t eltwise_next_rows     = 0; ///< SrcA and Dest on by 8 rows: the second half of a face
inline constexpr std::uint8_t eltwise_next_face_col = 1; ///< SrcA and Dest on by 8 rows: the next face in the face row
inline constexpr std::uint8_t eltwise_next_face_row = 2; ///< SrcA and Dest on by 8 rows: the first face of the next face row
inline constexpr std::uint8_t eltwise_pass_end      = 3; ///< SrcA and SrcB to 0; Dest by the walk: the 8 instructions of a pass are done
inline constexpr std::uint8_t eltwise_tile_end      = 4; ///< HiFi: SrcA and SrcB to 0, Dest and its CR on to the next tile, the fidelity phase to 0

/// Whether P's tiles are found by walking the Dest row counter. Not mul without .accumulate: it zeroes each tile
/// first, and ZEROACC takes its rows from the math Dest offset, so that one keeps an offset per tile.
constexpr bool eltwise_walks_dest(const eltwise_binary_params& params)
{
    return !(params.function == BinaryFunction::Mul && !params.accumulate);
}

/// The tile's instructions but the last: HiFi's last phase plays these, then the instruction that ends the tile.
inline constexpr hal::replay::BufferRange eltwise_tile_replay_but_last {eltwise_replay_start, eltwise_ops_per_tile - 1};

/// One ELWADD, ELWSUB or ELWMUL of P, then the counters move by address modifier AddrMod.
template <eltwise_binary_params P, std::uint8_t AddrMod, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void eltwise_instruction()
{
    hal::fpu::run<hal::fpu::Elementwise {
        .operation        = eltwise_fpu_operation(P.function),
        .broadcast        = eltwise_fpu_broadcast(P.broadcast),
        .dest_write       = hal::fpu::DestWriteMode::Overwrite, // Dest = SrcA op SrcB, not += (ELWMUL always adds: execute zeroes first)
        .address_modifier = AddrMod,
        .dest_row_offset  = 0, // the row comes from DEST_TARGET_REG_CFG_MATH_Offset and the Dest counter
        .release          = Release}>();
}

// ---- configure ------------------------------------------------------------------------------------------------------

/// Programs the math thread for P: Dest mode and ALU formats (math_common.h), the address modifiers, the replay of
/// one tile.
template <eltwise_binary_params P>
inline __attribute__((always_inline)) void eltwise_binary_configure()
{
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the address modifiers below assume four 16-row faces");
    static_assert(
        P.function == BinaryFunction::Add || P.function == BinaryFunction::Sub || P.function == BinaryFunction::Mul, "eltwise binary is add, sub or mul");
    static_assert(P.function == BinaryFunction::Mul || P.math_fidelity == 0, "ELWADD and ELWSUB run one fidelity phase (LoFi)");
    static_assert(P.function == BinaryFunction::Mul || !P.accumulate, "accumulate: mul only (ELWMUL adds to Dest)");
    static_assert(P.math_fidelity == 0 || (P.math_fidelity >= 2 && P.math_fidelity <= 4), "fidelity is LoFi (0) or HiFi2..4 (2..4 phases)");
    using namespace hal::counters::rwc;
    constexpr bool hifi = eltwise_hifi(P);

    configure_math_formats<P.dest_fp32, P.src_a_format, P.src_b_format>();

    // Address modifiers (counter_step: math_common.h). SrcA and Dest step 8 rows after every instruction; SrcB
    // as the file comment says, per broadcast.
    constexpr counter_step down_one_op {.incr = rows_per_fpu_op};
    constexpr counter_step row_to_face1 {.incr = P.face_r_dim};                              // Row: SrcB row 0 -> 16, face 1's row 0
    constexpr counter_step cr_to_face2 {.incr = faces_per_face_row * P.face_r_dim, .cr = 1}; // Col: SrcB (and its CR) to face 2's row 0
    constexpr counter_step b_rows     = P.broadcast == Broadcast::None || P.broadcast == Broadcast::Col ? down_one_op : counter_stay;
    constexpr counter_step b_face_col = P.broadcast == Broadcast::None  ? down_one_op
                                        : P.broadcast == Broadcast::Row ? row_to_face1
                                        : P.broadcast == Broadcast::Col ? counter_to_cr
                                                                        : counter_stay;
    constexpr counter_step b_face_row = P.broadcast == Broadcast::None  ? down_one_op
                                        : P.broadcast == Broadcast::Row ? counter_clear
                                        : P.broadcast == Broadcast::Col ? cr_to_face2
                                                                        : counter_stay;
    // eltwise_next_rows: second half of the face
    set_addr_mod<eltwise_next_rows, down_one_op, b_rows, down_one_op>();
    // eltwise_next_face_col: the next face in the face row
    set_addr_mod<eltwise_next_face_col, down_one_op, b_face_col, down_one_op>();
    // eltwise_next_face_row: the next face row
    set_addr_mod<eltwise_next_face_row, down_one_op, b_face_row, down_one_op>();
    // eltwise_pass_end: SrcA and SrcB to 0 and (HiFi) the fidelity phase on by one; Dest as the walk needs
    constexpr std::uint32_t next_fidelity_phase = hifi ? 1u : 0u;
    if constexpr (eltwise_walks_dest(P))
    {
        static_assert(dest_rows_per_tile(P) <= max_dest_counter_step, "the Dest step to the next tile does not fit an address modifier");
        // One phase: Dest steps on, to the next tile. HiFi: Dest back to the tile's first row (its CR) for the next
        // phase.
        set_addr_mod<eltwise_pass_end, counter_clear, counter_clear, hifi ? counter_to_cr : down_one_op, next_fidelity_phase>();
        if constexpr (hifi)
        {
            // eltwise_tile_end (after the last phase): Dest's CR (and Dest) on by one tile, the fidelity phase to 0.
            // One phase does not use the slot and leaves it alone.
            constexpr counter_step dest_next_tile {.incr = dest_rows_per_tile(P), .cr = 1};
            set_addr_mod<eltwise_tile_end, counter_clear, counter_clear, dest_next_tile, 0 /*FidelityIncr*/, 1 /*FidelityClear*/>();
        }
    }
    else
    {
        // Dest to 0: every tile has its own math Dest offset.
        set_addr_mod<eltwise_pass_end, counter_clear, counter_clear, counter_clear, next_fidelity_phase>();
    }
    set_hold_addr_mod(); // mul's ZEROACCs before each tile name it

    // One tile: faces 0, 1 (top), then 2, 3 (bottom), two instructions each. With one phase the last releases both
    // Src banks; with HiFi that waits until after the last phase.
    constexpr auto release = hifi ? hal::fpu::SourceRelease::None : hal::fpu::SourceRelease::Both;
    hal::replay::record<eltwise_tile_replay>(
        []
        {
            eltwise_instruction<P, eltwise_next_rows>();
            eltwise_instruction<P, eltwise_next_face_col>();
            eltwise_instruction<P, eltwise_next_rows>();
            eltwise_instruction<P, eltwise_next_face_row>();
            eltwise_instruction<P, eltwise_next_rows>();
            eltwise_instruction<P, eltwise_next_face_col>();
            eltwise_instruction<P, eltwise_next_rows>();
            eltwise_instruction<P, eltwise_pass_end, release>();
        });
    static_assert(eltwise_ops_per_tile == 8, "the replay above records 8 instructions: 4 faces x 2 instructions of 8 rows");

    reset_math_counters();
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// mul: marks the 4 faces of the output tile at Dest row tile_row (absolute) zero, so the ELWMULs' sums start from 0
/// (zero_dest_faces, math_common.h: the math Dest offset is left on the tile's Dest half; execute points it at the
/// tile next).
template <eltwise_binary_params P>
inline __attribute__((always_inline)) void eltwise_zero_tile(std::uint32_t tile_row)
{
    zero_dest_faces(tile_row, 0 /*first_face*/, P.num_faces, P.dest_fp32 ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16);
}

/// execute for a P whose tiles the Dest row counter walks (eltwise_walks_dest): add, sub, and mul with .accumulate.
/// The math Dest offset is written once, at the first tile.
template <eltwise_binary_params P, class Tiles>
inline __attribute__((always_inline)) void eltwise_execute_walking(std::uint32_t dest_row, const Tiles& tiles)
{
    // The math Dest offset once, at the first tile; the Dest row counter (0 after configure) walks the tiles.
    set_math_dest_tile<P>(dest_row, 0);
    for (std::uint32_t tile = 0; tile < wrapper::value(tiles); tile++)
    {
        if constexpr (eltwise_hifi(P))
        {
            // Every phase but the last from the replay; the last one's final instruction ends the tile instead
            // of the phase and hands both Src banks back.
#pragma GCC unroll 4
            for (std::uint32_t phase = 0; phase + 1 < eltwise_phases(P); phase++)
            {
                hal::replay::run<eltwise_tile_replay>();
            }
            hal::replay::run<eltwise_tile_replay_but_last>();
            eltwise_instruction<P, eltwise_tile_end, hal::fpu::SourceRelease::Both>();
        }
        else
        {
            hal::replay::run<eltwise_tile_replay>(); // its last instruction steps Dest on to the next tile
        }
    }
    // Dest and its CR back to 0: what runs next on this thread counts from the math Dest offset.
    hal::counters::set<hal::counters::rwc::MathCounter {.dest = hal::counters::rwc::Set(0)}>();
}

/// execute for mul without .accumulate: each tile is zeroed first, then written from its own math Dest offset.
/// With HiFi this form ends a tile differently from the walking one: every phase, the last included, comes from the
/// replay (eltwise_pass_end, which here also puts Dest back to 0), and one SETRWC (eltwise_release_both) then hands
/// both Src banks back and clears the fidelity phase. eltwise_tile_end is not programmed for this form.
template <eltwise_binary_params P, class Tiles>
inline __attribute__((always_inline)) void eltwise_execute_zeroing(std::uint32_t dest_row, const Tiles& tiles)
{
    for (std::uint32_t tile = 0; tile < wrapper::value(tiles); tile++)
    {
        eltwise_zero_tile<P>(dest_row + tile * dest_rows_per_tile(P)); // ELWMUL adds to Dest
        set_math_dest_tile<P>(dest_row, tile);
        // The 8 instructions once per fidelity phase, then (HiFi) both Src banks back to the unpackers.
        run_tile_replay<eltwise_tile_replay, eltwise_phases(P)>();
        if constexpr (eltwise_hifi(P))
        {
            hal::counters::set<eltwise_release_both>();
        }
    }
}

/// Every output tile, in the order the unpack walk delivers them (row-major), from dest_row (the absolute Dest row
/// the acc tensor starts at; compute_api/dest.h). tiles is their count: a ct (P.tile_rows * P.tile_cols), or an
/// rt of at most that.
template <eltwise_binary_params P, class Tiles, ctrt<Tiles, std::uint32_t> = 0>
inline __attribute__((always_inline)) void eltwise_binary_execute(std::uint32_t dest_row, const Tiles& tiles)
{
    if constexpr (wrapper::is_rt_v<Tiles>)
    {
        LLK_ASSERT(wrapper::value(tiles) >= 1 && wrapper::value(tiles) <= P.tile_rows * P.tile_cols, "a run-time tile count is outside 1..its bound");
    }
    if constexpr (eltwise_walks_dest(P))
    {
        eltwise_execute_walking<P>(dest_row, tiles);
    }
    else
    {
        eltwise_execute_zeroing<P>(dest_row, tiles);
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The math thread's part of a Compute API eltwise binary op: configure, then every output tile into Dest from
/// dest_row. counts is the run-time side of the Plan's Work (an empty object when every count is compile time).
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void eltwise_binary(const WorkCounts<Pl.work>& counts, std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    constexpr eltwise_binary_params params = eltwise_binary_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        eltwise_binary_configure<params>();
        eltwise_binary_execute<params>(dest_row, plan_count<WorkCount::Tiles, Pl.work.tile_rows * Pl.work.tile_cols>(counts));
    }
}

} // namespace llk2::llk_lib
