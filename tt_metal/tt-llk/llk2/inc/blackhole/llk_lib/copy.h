// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, copy on Blackhole. Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/), except where
// noted. The math thread's part of a Compute API copy; the unpackers' part is unpack.h. The shape of an op:
// llk_lib/thread.h.
//
//   copy_params                 one struct; copy_configure and copy_execute take the same one
//   copy_params_from(Plan)      fills it from the Resolver's Plan
//   copy_configure<P>()         through SrcA: Dest mode, ALU formats and the Src zero flag (math_common.h), the
//                               address modifiers, the row counters. Straight to Dest: the address modifier its
//                               ZEROACCs name, the row counters
//   copy_execute<P>(dest_row)   every tile into Dest from dest_row. Straight to Dest: the hand-off with the unpackers,
//                               then the zero-flag clears
//   copy<On, Plan>(counts, dest_row)
//                               the per-thread call the Compute API makes: configures and executes, only when this
//                               build is for thread On (the math thread). counts (WorkCounts) is the run-time side
//                               of the Plan's Work, and is empty: copy takes no run-time size
//
// Through SrcA. The unpacker writes one whole tile into a SrcA bank (64 rows: faces 0, 1, 2, 3, 16 rows each) and
// hands the bank over. Math points its Dest offset at the tile's first row and moves the 64 rows with MOVA2D, eight
// rows per instruction; between them an address modifier (copy_next_rows) moves the SrcA and Dest counters on by
// eight rows, and after the last one another (copy_rows_done) puts them back to 0. Then math flips SrcA, which gives
// the bank back to the unpacker. MOVA2D writes 16-bit datums to a 16-bit Dest and widens to 32 bits when Dest is
// 32-bit (on Blackhole, ALU_ACC_CTRL_Fp32_enabled makes MOVA2D treat SrcA as TF32; ISA: MOVA2D). With transpose,
// the unpacker has already transposed inside each face; math moves Src face f to Dest face copy_dest_face(f), which
// swaps faces 1 and 2.
//
// Straight to Dest (unpack.h, "Unpack to Dest"). Math tells the unpacker where to write and waits until it has:
//   1. wait until math's own earlier Dest writes are done (semaphore MATH_DONE, math_done_sync: posted from the math
//      pipe once it is idle, then taken back by the RISC-V)
//   2. send acc's Dest row (absolute: a Dest ring's tensors start at their section's row) through the mailbox.
//      Math owns the section by now (the ring's reserve waited on math); the unpack thread's reserve does not wait,
//      so this mailbox is what keeps the unpacker out of a section the packer still reads
//   3. stall math until the unpacker posts UNPACK_TO_DEST, then take it
//   4. mark every written face's rows as written (ZEROACC clear-zero-flags): on Blackhole, an unpack to Dest in the
//      same cycle as a packer ZEROACC loses its zero-flag clear.
//      A face ZEROACC applies its address modifier, and Blackhole picks the Dest half it works in from offset + index
//      + the Dest row counter (tt-isa-documentation, ZEROACC.md). So configure programs a slot that moves nothing
//      (addr_mod_hold, math_common.h) and resets the counters, as eltwise_binary.h and reduce.h do for their ZEROACCs: with a slot left
//      as "Dest on by eight rows" by an earlier op, the counter would move on with every face, and after enough
//      copies the clears would land in the other half (test_llk2_copy_to_dest_repeat).
//
// Departures from the shape (llk_lib/thread.h), straight to Dest only, and why:
//   - the op takes Dest over inside execute, by its own protocol with the unpack thread (the two semaphores and the
//     mailbox above) and not through sync.h. Forced by the hardware: the unpacker writes Dest from the unpack thread,
//     and only math can tell when the FPU and SFPU have left Dest; sync.h has no unpack <-> math hand-off.
//
// Ported from: llk_math_eltwise_unary_datacopy.h (A2D, no broadcast; its unpack-to-Dest handshake) and the face
// reorder of transpose_wh_tile.
//
// Not HAL: nothing.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h" // configure_math_formats, set_addr_mod, dest_half_first_row
#include "llk_lib/platform/mailbox.h"
#include "llk_lib/semaphores.h" // math_done_sync, unpack_to_dest_sync
#include "llk_lib/thread.h"
#include "resolver/work_counts.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

// ---- parameters -----------------------------------------------------------------------------------------------------

/// Parameters of copy on the math thread. configure and execute take the same struct.
struct copy_params
{
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tiles;      ///< tiles to copy, in L1 order, to consecutive Dest tiles
    bool to_dest;             ///< the unpacker writes Dest itself; math only hands off
    bool transpose;           ///< move the faces in transposed order (the unpacker transposed inside each face)
    bool dest_fp32;           ///< 32-bit Dest mode
    DataFormat src_a_format;  ///< the format in SrcA, which MOVA2D reads
};

/// The Plan of a copy as math parameters.
constexpr copy_params copy_params_from(const Plan& plan)
{
    return {
        .face_r_dim   = plan.work.face_r_dim,
        .num_faces    = plan.work.num_faces,
        .tiles        = plan.work.tile_rows * plan.work.tile_cols,
        .to_dest      = plan.in0.to == Where::Dest,
        .transpose    = plan.math.transpose_faces,
        .dest_fp32    = plan.math.dest_fp32,
        .src_a_format = plan.math.format,
    };
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

inline constexpr std::uint8_t copy_next_rows = 0; ///< address modifier slot: SrcA and Dest counters on by one MOVA2D (eight rows)
inline constexpr std::uint8_t copy_rows_done = 1; ///< address modifier slot: SrcA and Dest counters back to 0, after a transfer's last MOVA2D

/// The Dest face that Src face f goes to: itself, or with transpose the face mirrored over the diagonal
/// (face (r, c) -> face (c, r): 1 <-> 2).
constexpr std::uint32_t copy_dest_face(const copy_params& params, std::uint32_t f)
{
    return params.transpose ? (f % faces_per_face_row) * faces_per_face_row + f / faces_per_face_row : f;
}

/// Moves Rows rows from SrcA row Src to Dest row Dst (relative to the Dest offset). A transfer is several MOVA2Ds
/// of eight rows; the counters carry each on to the next eight and are back at 0 after the last.
template <std::uint32_t Src, std::uint32_t Dst, hal::move::RowCount Rows>
inline __attribute__((always_inline)) void copy_move_rows()
{
    hal::move::run<hal::move::Transfer<hal::move::SrcA, hal::move::Dst> {
        .source_row = Src, .destination_row = Dst, .address_mode = copy_next_rows, .number_of_rows = Rows, .final_address_mode = copy_rows_done}>();
}

/// Moves Src face F of the current SrcA bank to its Dest face.
template <copy_params P, std::uint32_t F>
inline __attribute__((always_inline)) void copy_move_face()
{
    copy_move_rows<F * P.face_r_dim, copy_dest_face(P, F) * P.face_r_dim, hal::move::rows::Face>();
}

// ---- configure ------------------------------------------------------------------------------------------------------

/// Programs the math thread for P: Dest mode and ALU formats (math_common.h), the address modifiers execute's
/// instructions name, the row counters. Straight to Dest the FPU moves no data: the instructions are the ZEROACCs.
template <copy_params P>
inline __attribute__((always_inline)) void copy_configure()
{
    static_assert(P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now");
    static_assert(!(P.to_dest && P.transpose), "the Resolver never transposes on the way to Dest");
    // Dest mode, ALU format baseline, implied SrcA format, Src bank release and Src zero flag (the math side of
    // hw_configure). SrcB is unused.
    configure_math_formats<P.dest_fp32, P.src_a_format, P.src_a_format>();
    if constexpr (!P.to_dest)
    {
        // Address modifiers: on by eight rows between a transfer's MOVA2Ds, back to 0 after its last.
        set_addr_mod<copy_next_rows, counter_next_fpu_op, counter_stay, counter_next_fpu_op>();
        set_addr_mod<copy_rows_done, counter_clear, counter_stay, counter_clear>();
        reset_math_counters();
    }
    else
    {
        // The face ZEROACCs apply their address modifier and take their Dest half from the Dest row counter: the
        // slot that moves nothing, and the counters at 0.
        set_hold_addr_mod();
        reset_math_counters();
    }
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// Through SrcA: each tile from its SrcA bank to Dest, then the bank back to the unpacker.
template <copy_params P>
inline __attribute__((always_inline)) void copy_through_src(std::uint32_t dest_row)
{
    using namespace hal::counters::rwc;
    constexpr MathCounter release_src_a {.src_a = Set(0), .dest = Set(0), .flip = Src::A};
    for (std::uint32_t tile = 0; tile < P.tiles; tile++)
    {
        set_math_dest_tile<P>(dest_row, tile);
        if constexpr (!P.transpose)
        {
            // The whole tile in one transfer: 64 rows, eight MOVA2Ds of eight rows.
            copy_move_rows<0, 0, hal::move::rows::faces<P.num_faces>()>();
        }
        else
        {
            copy_move_face<P, 0>();
            copy_move_face<P, 1>();
            copy_move_face<P, 2>();
            copy_move_face<P, 3>();
        }
        set<release_src_a>();
    }
}

/// Straight to Dest, after the unpacker has written: marks every written face's rows as written (step 4 of the file
/// comment), through zero_dest_faces (math_common.h), which leaves the math Dest offset on the tensor's Dest half.
template <copy_params P>
inline __attribute__((always_inline)) void copy_clear_zero_flags(std::uint32_t dest_row)
{
    // Unpack to Dest is 32-bit only (resolver/copy.h).
    zero_dest_faces(dest_row, 0 /*first_face*/, P.tiles * P.num_faces, hal::dst::DestWidth::Bits32, hal::dst::ZeroFlagAction::ClearFlags);
}

/// Straight to Dest: the hand-off with the unpacker (steps 1 to 3 of the file comment; unpack.h,
/// unpack_dest_write_begin / unpack_dest_write_end).
inline __attribute__((always_inline)) void copy_to_dest_handoff(std::uint32_t dest_row)
{
    // 1. Math's earlier Dest writes are done: the math pipe posts MATH_DONE once math and SFPU are idle, and the
    //    RISC-V waits to see it and takes it back.
    hal::sync::wait::semaphore<StallTarget::Sync, SemaphoreCondition::WhileMaximum, math_done_sync>();
    hal::sync::wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    hal::sync::semaphore::post<hal::sync::Access::Tensix, math_done_sync>();
    while (hal::sync::semaphore::read<math_done_sync>() == 0)
    {
    }
    hal::sync::semaphore::get<hal::sync::Access::MMIO, math_done_sync>();

    // 2. Where to write.
    platform::mailbox::write(platform::mailbox::Thread::Unpack, dest_row);

    // 3. Math (and anything after it on this thread) waits until the unpacker has written every tile.
    constexpr StallTarget math_pipe = StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync;
    hal::sync::wait::semaphore<math_pipe, SemaphoreCondition::WhileZero, unpack_to_dest_sync>();
    hal::sync::semaphore::get<hal::sync::Access::Tensix, unpack_to_dest_sync>();
}

/// Every tile of the call into Dest from dest_row (absolute).
template <copy_params P>
inline __attribute__((always_inline)) void copy_execute(std::uint32_t dest_row)
{
    if constexpr (P.to_dest)
    {
        copy_to_dest_handoff(dest_row);
        copy_clear_zero_flags<P>(dest_row);
    }
    else
    {
        copy_through_src<P>(dest_row);
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The math thread's part of a Compute API copy: configure, then every tile into Dest from dest_row. counts is the
/// run-time side of the Plan's Work, an empty object: copy takes no run-time size.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void copy(const WorkCounts<Pl.work>&, std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    static_assert(
        Pl.work.rows_from == CountSource::Fixed && Pl.work.cols_from == CountSource::Fixed && Pl.work.k_from == CountSource::Fixed,
        "copy takes no run-time size: every count of its Work is compile time");
    constexpr copy_params params = copy_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        copy_configure<params>();
        copy_execute<params>(dest_row);
    }
}

} // namespace llk2::llk_lib
