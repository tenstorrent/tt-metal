// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, copy on Blackhole. Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/), except where
// noted. Math only: the unpack side of copy is in unpack.h. Ports llk_math_eltwise_unary_datacopy.h (A2D, no
// broadcast) and the face reorder of today's transpose_wh_tile.
//
//   copy_params              one struct; copy_configure and copy_execute take the same one
//   copy_params_from(Plan)   fills it from the Resolver's Plan
//   copy_configure<P>()      through SrcA: Dest mode, ALU formats and the Src zero flag (math_common.h), the
//                            address modifiers. Straight to Dest: nothing; math only waits
//   copy_execute<P>(row)     every tile into Dest from row
//   copy<On, Plan>(row)      the per-thread call the Compute API makes: configures and executes, only when this
//                            build is for thread On (the math thread)
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
//      same cycle as a packer ZEROACC loses its zero-flag clear (budabackend#2730)
//
// Not HAL yet: the thread mailboxes (mailbox_write, ckernel.h); the Src zero-flag rule is in math_common.h.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/dest.h"        // dest_section_rows
#include "llk_lib/math_common.h" // configure_math_formats, set_addr_mod
#include "llk_lib/semaphores.h"  // math_done_sync, unpack_to_dest_sync
#include "llk_lib/thread.h"
#include "resolver/plan.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// Parameters of copy on the math thread. configure and execute take the same struct.
struct copy_params
{
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tiles;      ///< tiles to copy, in L1 order, to consecutive Dest tiles
    bool to_dest;             ///< the unpacker writes Dest itself; math only hands off
    bool transpose;           ///< move the faces in transposed order (the unpacker transposed inside each face)
    bool dest_fp32;           ///< 32-bit Dest mode
    DataFormat math_format;   ///< the SrcA format MOVA2D reads
};

/// The Plan of a copy as math parameters.
constexpr copy_params copy_params_from(const Plan& p)
{
    return {
        p.work.face_r_dim,
        p.work.num_faces,
        p.work.rt_dim * p.work.ct_dim,
        p.in0.to == Where::Dest,
        p.math.transpose_faces,
        p.math.dest_fp32,
        p.math.format,
    };
}

inline constexpr std::uint8_t copy_next_rows = 0; ///< address modifier slot: SrcA and Dest counters on by one MOVA2D (eight rows)
inline constexpr std::uint8_t copy_rows_done = 1; ///< address modifier slot: SrcA and Dest counters back to 0, after a transfer's last MOVA2D

/// The Dest face that Src face f goes to: itself, or with transpose the face mirrored over the diagonal
/// (face (r, c) -> face (c, r): 1 <-> 2).
constexpr std::uint32_t copy_dest_face(const copy_params& p, std::uint32_t f)
{
    return p.transpose ? (f % faces_per_face_row) * faces_per_face_row + f / faces_per_face_row : f;
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

/// Programs the math thread for P. Straight to Dest needs nothing: math issues no FPU work.
template <copy_params P>
inline __attribute__((always_inline)) void copy_configure()
{
    static_assert(P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now");
    static_assert(!(P.to_dest && P.transpose), "the Resolver never transposes on the way to Dest");
    if constexpr (!P.to_dest)
    {
        // Dest mode, ALU format baseline, implied SrcA format, Src bank release and Src zero flag (the math side of
        // hw_configure). SrcB is unused.
        configure_math_formats<P.dest_fp32, P.math_format, P.math_format>();

        // Address modifiers: on by eight rows between a transfer's MOVA2Ds, back to 0 after its last.
        constexpr counter_step next_eight {rows_per_fpu_op, 0, 0};
        set_addr_mod<static_cast<cfg::Sec>(copy_next_rows), next_eight, counter_stay, next_eight>();
        set_addr_mod<static_cast<cfg::Sec>(copy_rows_done), counter_clear, counter_stay, counter_clear>();
        reset_math_counters();
    }
}

/// Through SrcA: each tile from its SrcA bank to Dest, then the bank back to the unpacker.
template <copy_params P>
inline __attribute__((always_inline)) void copy_through_src(std::uint32_t dest_row)
{
    using namespace hal::counters::rwc;
    constexpr MathCounter release_src_a {.src_a = Set(0), .dest = Set(0), .flip = Src::A};
    for (std::uint32_t tile = 0; tile < P.tiles; tile++)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row + tile * dest_rows_per_tile(P));
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

/// Straight to Dest: the handoff with the unpacker (unpack.h, dest_write_begin / dest_write_end).
template <copy_params P>
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

    // 2. Where to write. Not HAL: the thread mailboxes.
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, dest_row);

    // 3. Math (and anything after it on this thread) waits until the unpacker has written every tile.
    constexpr StallTarget math_pipe = StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync;
    hal::sync::wait::semaphore<math_pipe, SemaphoreCondition::WhileZero, unpack_to_dest_sync>();
    hal::sync::semaphore::get<hal::sync::Access::Tensix, unpack_to_dest_sync>();

    // 4. Zero-flag clear per face (budabackend#2730). ZEROACC's face index counts 16-row faces from the start of the
    //    Dest section the math offset selects, so the offset goes to the start of acc's section first.
    const std::uint32_t section_row = dest_row - dest_row % dest_section_rows;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(section_row);
    const std::uint32_t first_face = (dest_row - section_row) / P.face_r_dim;
    for (std::uint32_t face = 0; face < P.tiles * P.num_faces; face++)
    {
        hal::dst::run(hal::dst::Zero {
            .scope = hal::dst::ZeroScope::Face,
            .index = first_face + face,
            .flags = hal::dst::ZeroFlagAction::ClearFlags,
            .width = hal::dst::DestWidth::Bits32}); // unpack to Dest is 32-bit only (resolver/copy.h)
    }
}

/// Every tile of the call into Dest from dest_row (absolute).
template <copy_params P>
inline __attribute__((always_inline)) void copy_execute(std::uint32_t dest_row)
{
    if constexpr (P.to_dest)
    {
        copy_to_dest_handoff<P>(dest_row);
    }
    else
    {
        copy_through_src<P>(dest_row);
    }
}

/// The math thread's part of a Compute API copy: configure, then every tile into Dest from dest_row.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void copy(std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "copy's math side runs on the math thread");
    constexpr copy_params p = copy_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        copy_configure<p>();
        copy_execute<p>(dest_row);
    }
}

} // namespace llk2::llk_lib
