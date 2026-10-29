// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, eltwise binary (add, sub) on Blackhole. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/). Math only: the unpack side is the shared unpack.h. Ported from
// tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h (the standard path: no Dest reuse, no accumulation).
//
//   eltwise_binary_params                 one struct; configure and execute take the same one
//   eltwise_binary_params_from(Plan)      fills it from the Resolver's Plan
//   eltwise_binary_configure<P>()         Dest mode and ALU formats, the address modifiers, the replay of one tile's
//                                         eight FPU instructions and the MOP that runs it
//   eltwise_binary_execute<P>(row)        every output tile: Dest = SrcA <function> SrcB, from Dest row `row`
//   eltwise_binary<On, Plan>(row)         the per-thread call the Compute API makes: configures and executes, only
//                                         when this build is for thread On (the math thread)
//
// How one tile is computed. The unpackers hand math one whole tile per Src bank: SrcA rows 0..63 hold in0's faces
// 0..3. An ELWADD (ELWSUB) takes 8 rows of SrcA and 8 rows of SrcB and writes 8 rows of Dest, so a 32x32 tile is
// 8 instructions. They run from a replay buffer; between them, the address modifiers move the row counters:
//   elw_rows       the second half of a face
//   elw_face_col   the next face in the face row (face 0 -> 1, 2 -> 3)
//   elw_face_row   the first face of the next face row (face 1 -> 2)
//   elw_done       every counter back to 0
// The first three all move SrcA and Dest on by 8 rows; they differ only in SrcB. SrcB moves with SrcA when it is a
// whole tile. With a row broadcast, SrcB holds only in1's top face row (rows 0..15
// face 0, rows 16..31 face 1) and the instruction reads the one SrcB row its counter selects and repeats it down all
// 8 rows. So SrcB stays on row 0 through face 0, moves to row 16 (face 1's row 0) for face 1, and back to row 0 and
// 16 for faces 2 and 3. The last instruction of the tile releases both Src banks to the unpackers.
//
// Shared math configure (and its non-HAL parts): math_common.h.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h"
#include "llk_lib/thread.h"
#include "resolver/plan.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;

/// Parameters of an eltwise binary op on the math thread. configure and execute take the same struct.
struct eltwise_binary_params
{
    std::uint32_t face_r_dim; ///< rows per face, both operands
    std::uint32_t num_faces;  ///< faces per tile, both operands
    std::uint32_t rt_dim;     ///< output tiles along y
    std::uint32_t ct_dim;     ///< output tiles along x
    BinaryFunction function;  ///< add or sub
    Broadcast broadcast;      ///< how SrcB is spread over SrcA
    bool dest_fp32;           ///< 32-bit Dest mode
    DataFormat src_a_format;  ///< the format in SrcA
    DataFormat src_b_format;  ///< the format in SrcB
};

/// The operand of a Plan that the unpackers write to `where`.
constexpr const UnpackPlan& eltwise_binary_operand_in(const Plan& p, Where where)
{
    return p.in0.to == where ? p.in0 : p.in1;
}

/// The Plan of an eltwise binary op as math parameters.
constexpr eltwise_binary_params eltwise_binary_params_from(const Plan& p)
{
    return {
        p.work.face_r_dim,
        p.work.num_faces,
        p.work.rt_dim,
        p.work.ct_dim,
        static_cast<BinaryFunction>(p.math.function),
        eltwise_binary_operand_in(p, Where::SrcB).broadcast,
        p.math.dest_fp32,
        eltwise_binary_operand_in(p, Where::SrcA).writes,
        eltwise_binary_operand_in(p, Where::SrcB).writes,
    };
}

inline constexpr std::uint32_t elw_ops_per_tile = faces_per_full_tile * (max_face_rows / rows_per_fpu_op); ///< 4 faces x 2 instructions of 8 rows

/// The FPU instruction for a BinaryFunction: ELWADD or ELWSUB.
constexpr hal::fpu::ElementwiseOperation elw_fpu_operation(BinaryFunction f)
{
    return f == BinaryFunction::Sub ? hal::fpu::ElementwiseOperation::Subtract : hal::fpu::ElementwiseOperation::Add;
}

/// The FPU's SrcB broadcast for an operand's Broadcast.
constexpr hal::fpu::SrcBBroadcast elw_fpu_broadcast(Broadcast b)
{
    return b == Broadcast::Row ? hal::fpu::SrcBBroadcast::Row : hal::fpu::SrcBBroadcast::None;
}

// The address modifier slots the replay uses (ADDR_MOD_0..3, as the original LLK).
inline constexpr std::uint8_t elw_rows     = 0; ///< SrcA and Dest on by 8 rows: the second half of a face
inline constexpr std::uint8_t elw_face_col = 1; ///< SrcA and Dest on by 8 rows: the next face in the face row
inline constexpr std::uint8_t elw_face_row = 2; ///< SrcA and Dest on by 8 rows: the first face of the next face row
inline constexpr std::uint8_t elw_done     = 3; ///< every counter to 0: the tile is done

/// One ELWADD or ELWSUB of P, then the counters move by address modifier AddrMod.
template <eltwise_binary_params P, std::uint8_t AddrMod, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void elw_instruction()
{
    hal::fpu::run<hal::fpu::Elementwise {
        .operation        = elw_fpu_operation(P.function),
        .broadcast        = elw_fpu_broadcast(P.broadcast),
        .dest_write       = hal::fpu::DestWriteMode::Overwrite, // Dest = SrcA op SrcB, not +=
        .address_modifier = AddrMod,
        .dest_row_offset  = 0, // the row comes from DEST_TARGET_REG_CFG_MATH_Offset and the Dest counter
        .release          = Release}>();
}

/// Programs the math thread for P: Dest mode and ALU formats (math_common.h), address modifiers, the replay and the MOP.
template <eltwise_binary_params P>
inline __attribute__((always_inline)) void eltwise_binary_configure()
{
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the address modifiers below assume four 16-row faces");
    static_assert(P.function == BinaryFunction::Add || P.function == BinaryFunction::Sub, "only add and sub are ported: mul needs fidelity phases");
    static_assert(P.broadcast == Broadcast::None || P.broadcast == Broadcast::Row, "column and scalar broadcast are not ported yet");

    configure_math_formats<P.dest_fp32, P.src_a_format, P.src_b_format>();

    // Address modifiers (counter_step: math_common.h). SrcA and Dest step 8 rows after every instruction; SrcB
    // either does the same (a whole tile) or walks row 0 of faces 0 and 1 (row broadcast).
    constexpr bool row_bcast = P.broadcast == Broadcast::Row;
    constexpr counter_step down_one_op {rows_per_fpu_op, 0, 0};
    constexpr counter_step to_next_face_row0 {P.face_r_dim, 0, 0}; // SrcB row 0 -> 16: face 1's row 0
    // elw_rows: second half of the face; a broadcast row stays put
    set_addr_mod<static_cast<cfg::Sec>(elw_rows), down_one_op, row_bcast ? counter_stay : down_one_op, down_one_op>();
    // elw_face_col: the next face in the face row
    set_addr_mod<static_cast<cfg::Sec>(elw_face_col), down_one_op, row_bcast ? to_next_face_row0 : down_one_op, down_one_op>();
    // elw_face_row: the next face row; a broadcast row goes back to face 0's row 0
    set_addr_mod<static_cast<cfg::Sec>(elw_face_row), down_one_op, row_bcast ? counter_clear : down_one_op, down_one_op>();
    // elw_done: every counter to 0 for the next tile
    set_addr_mod<static_cast<cfg::Sec>(elw_done), counter_clear, counter_clear, counter_clear>();

    // One tile: faces 0, 1 (top), then 2, 3 (bottom), two instructions each. The last releases both Src banks.
    constexpr hal::replay::BufferRange tile_replay {math_replay_tile_start, elw_ops_per_tile};
    hal::replay::record<tile_replay>(
        []
        {
            elw_instruction<P, elw_rows>();
            elw_instruction<P, elw_face_col>();
            elw_instruction<P, elw_rows>();
            elw_instruction<P, elw_face_row>();
            elw_instruction<P, elw_rows>();
            elw_instruction<P, elw_face_col>();
            elw_instruction<P, elw_rows>();
            elw_instruction<P, elw_done, hal::fpu::SourceRelease::Both>();
        });
    static_assert(elw_ops_per_tile == 8, "the replay above records 8 instructions: 4 faces x 2 instructions of 8 rows");

    // MOP template 1: the tile replay once (one pass of the outer loop, one of the inner).
    hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
        .outer_loop = {.count = 1}, .inner_loop = {.count = 1, .body_op = hal::replay::get_operation<tile_replay>()}}>();
    reset_math_counters();
}

/// Every output tile, in the order the unpack walk delivers them (row-major), from dest_row (the absolute Dest row
/// the acc tensor starts at; compute_api/dest.h).
template <eltwise_binary_params P>
inline __attribute__((always_inline)) void eltwise_binary_execute(std::uint32_t dest_row)
{
    for (std::uint32_t tile = 0; tile < P.rt_dim * P.ct_dim; tile++)
    {
        set_math_dest_tile<P>(dest_row, tile);
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
}

/// The math thread's part of a Compute API eltwise binary op: configure, then every output tile into Dest from dest_row.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void eltwise_binary(std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    constexpr eltwise_binary_params p = eltwise_binary_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        eltwise_binary_configure<p>();
        eltwise_binary_execute<p>(dest_row);
    }
}

} // namespace llk2::llk_lib
