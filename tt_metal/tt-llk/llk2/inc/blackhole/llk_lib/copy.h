// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, copy on Blackhole. Written in HAL calls (tt_llk_blackhole/llk_lib/hal/). Math only: the
// unpack side of copy is in unpack.h. Ports llk_math_eltwise_unary_datacopy.h (A2D, no broadcast) and the face
// reorder of today's transpose_wh_tile.
//
//   copy_params              one struct; copy_configure and copy_execute take the same one
//   copy_params_from(Plan)   fills it from the Resolver's Plan
//   copy_configure<P>()      through SrcA: Dest mode and ALU formats, the address modifier, the Src zero flag.
//                            Straight to Dest: nothing; math only waits
//   copy_execute<P>(row)     every tile into Dest from row
//   copy<On, Plan>(row)      the per-thread call the Compute API makes: configures and executes, only when this
//                            build is for thread On (the math thread)
//
// Through SrcA. The unpacker writes one whole tile into a SrcA bank (64 rows: faces 0, 1, 2, 3, 16 rows each) and
// hands the bank over. Math points its Dest offset at the tile's first row and moves the 64 rows with MOVA2D, eight
// rows per instruction; between them an address modifier (copy_next_rows) moves the SrcA and Dest counters on by
// eight rows, and after the last one another (copy_rows_done) puts them back to 0. Then math flips SrcA, which gives
// the bank back to the unpacker. MOVA2D writes 16-bit datums to a
// 16-bit Dest and widens to 32 bits when Dest is 32-bit (on Blackhole, ALU_ACC_CTRL_Fp32_enabled makes MOVA2D treat
// SrcA as TF32; ISA: MOVA2D). With transpose, the unpacker has already transposed inside each face; math moves Src
// face f to Dest face transposed_face(f), which swaps faces 1 and 2.
//
// Straight to Dest (unpack.h, "Unpack to Dest"). Math tells the unpacker where to write and waits until it has:
//   1. wait until math's own earlier Dest writes are done (semaphore MATH_DONE, posted from the math pipe once it
//      is idle, then taken back by the RISC-V)
//   2. send the absolute Dest row (row + the current Dest section base) through the mailbox
//   3. stall math until the unpacker posts UNPACK_TO_DEST, then take it
//   4. mark every written face's rows as written (ZEROACC clear-zero-flags): on Blackhole, an unpack to Dest in the
//      same cycle as a packer ZEROACC loses its zero-flag clear (budabackend#2730)
//
// Not HAL yet: the Dest section base (get_dest_buffer_base, ckernel.h), the Src zero-flag rule
// (requires_disabled_src_zero_flag, cmath_common.h) and the thread mailboxes (mailbox_write, ckernel.h).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/thread.h"
#include "llk_lib/unpack.h" // unpack_to_dest_sync: the semaphore of the unpack-to-Dest handoff
#include "resolver/plan.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::Semaphore;
using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// Parameters of copy on the math thread. configure and execute take the same struct.
struct copy_params
{
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tiles;      ///< tiles to copy, in L1 order, to consecutive Dest tiles
    bool unpack_to_dest;      ///< the unpacker writes Dest itself; math only hands off
    bool transpose;           ///< move the faces in transposed order (the unpacker transposed inside each face)
    bool dest_fp32;           ///< 32-bit Dest mode
    DataFormat math_format;   ///< the SrcA format MOVA2D reads
};

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

inline constexpr std::uint8_t copy_next_rows  = 0;             ///< address modifier slot: SrcA and Dest counters on by one MOVA2D (eight rows)
inline constexpr std::uint8_t copy_rows_done  = 1;             ///< address modifier slot: SrcA and Dest counters back to 0, after a transfer's last MOVA2D
inline constexpr std::uint32_t mova2d_rows    = 8;             ///< rows one MOVA2D moves (its eight-row form)
inline constexpr Semaphore math_done_sync     = Semaphore::S7; ///< ckernel::semaphore::MATH_DONE: math idle before an unpack to Dest
inline constexpr std::uint32_t faces_per_side = 2;             ///< a 32x32 tile is 2 x 2 faces, stored face 0, 1 (top), 2, 3 (bottom)

/// Dest rows one tile takes.
constexpr std::uint32_t copy_dest_rows_per_tile(const copy_params& p)
{
    return p.num_faces * p.face_r_dim;
}

/// The Dest face that Src face f goes to: itself, or with transpose the face mirrored over the diagonal
/// (face (r, c) -> face (c, r): 1 <-> 2).
constexpr std::uint32_t copy_dest_face(const copy_params& p, std::uint32_t f)
{
    return p.transpose ? (f % faces_per_side) * faces_per_side + f / faces_per_side : f;
}

/// Moves Rows rows from SrcA row Src to Dest row Dst (relative to the Dest offset). A transfer is several MOVA2Ds
/// of eight rows; the counters carry each on to the next eight and are back at 0 after the last.
template <std::uint32_t Src, std::uint32_t Dst, hal::move::RowCount Rows>
inline __attribute__((always_inline)) void move_rows()
{
    hal::move::run<hal::move::Transfer<hal::move::SrcA, hal::move::Dst> {
        .source_row = Src, .destination_row = Dst, .address_mode = copy_next_rows, .number_of_rows = Rows, .final_address_mode = copy_rows_done}>();
}

/// Moves Src face F of the current SrcA bank to its Dest face.
template <copy_params P, std::uint32_t F>
inline __attribute__((always_inline)) void move_face()
{
    move_rows<F * P.face_r_dim, copy_dest_face(P, F) * P.face_r_dim, hal::move::rows::Face>();
}

/// Programs one address modifier slot for SrcA and Dest; SrcB, fidelity and bias stay put.
template <std::uint8_t Slot, std::uint32_t Incr, std::uint32_t Clear>
inline __attribute__((always_inline)) void set_copy_addr_mod()
{
    using cfg::AddrMod;
    constexpr cfg::Sec slot = static_cast<cfg::Sec>(Slot);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::SrcA].Incr, slot, Incr>(),
        cfg::set<AddrMod[cfg::SrcA].Clear, slot, Clear>(),
        cfg::set<AddrMod[cfg::SrcA].CR, slot, 0>(),
        cfg::set<AddrMod[cfg::SrcB].Incr, slot, 0>(),
        cfg::set<AddrMod[cfg::SrcB].Clear, slot, 0>(),
        cfg::set<AddrMod[cfg::SrcB].CR, slot, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::Dest].Incr, slot, Incr>(),
        cfg::set<AddrMod[cfg::Dest].Clear, slot, Clear>(),
        cfg::set<AddrMod[cfg::Dest].CR, slot, 0>(),
        cfg::set<AddrMod[cfg::Fidelity].Incr, slot, 0>(),
        cfg::set<AddrMod[cfg::Fidelity].Clear, slot, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, AddrMod[cfg::Bias].Incr, slot, 0>();
}

/// Programs the math thread for P. Straight to Dest needs nothing: math issues no FPU work.
template <copy_params P>
inline __attribute__((always_inline)) void copy_configure()
{
    using namespace ckernel;
    using namespace hal::counters::rwc;
    static_assert(P.num_faces == 4 && P.face_r_dim == FACE_R_DIM, "32x32 tiles only, for now");
    static_assert(!(P.unpack_to_dest && P.transpose), "the Resolver never transposes on the way to Dest");
    if constexpr (!P.unpack_to_dest)
    {
        // Dest mode and ALU format baseline (the math side of hw_configure). Wait for math and SFPU to drain first.
        hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, 0>(); // float data only
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
            cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
            cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
            cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());
        // SrcA's format comes from the bank the unpacker filled (Blackhole's implied format), and flipping SrcA hands
        // the bank back to the unpacker. Each is one whole thread-config word.
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

        // Address modifiers: on by eight rows between a transfer's MOVA2Ds, back to 0 after its last.
        set_copy_addr_mod<copy_next_rows, mova2d_rows, 0>();
        set_copy_addr_mod<copy_rows_done, 0, 1>();
        set<MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .clear_fidelity = true}>();

        // Src zero-flag baseline for this format; the ckernel caches stay in step for the LLK asserts.
        const std::uint32_t math_format = static_cast<std::uint32_t>(P.math_format);
        math::src_zero_flag_srca_fmt    = math_format;
        math::src_zero_flag_srcb_fmt    = math_format;
        const std::uint32_t zero_flag   = requires_disabled_src_zero_flag(math_format, math_format) ? 1u : 0u;
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(zero_flag);
        math::src_zero_flag_hw = zero_flag;
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
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(
            dest_row + tile * copy_dest_rows_per_tile(P) + ckernel::get_dest_buffer_base());
        if constexpr (!P.transpose)
        {
            // The whole tile in one transfer: 64 rows, eight MOVA2Ds of eight rows.
            move_rows<0, 0, hal::move::rows::faces<P.num_faces>()>();
        }
        else
        {
            move_face<P, 0>();
            move_face<P, 1>();
            move_face<P, 2>();
            move_face<P, 3>();
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
    const std::uint32_t section_base = ckernel::get_dest_buffer_base();
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, section_base + dest_row);

    // 3. Math (and anything after it on this thread) waits until the unpacker has written every tile.
    constexpr StallTarget math_pipe = StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync;
    hal::sync::wait::semaphore<math_pipe, SemaphoreCondition::WhileZero, unpack_to_dest_sync>();
    hal::sync::semaphore::get<hal::sync::Access::Tensix, unpack_to_dest_sync>();

    // 4. Zero-flag clear per face (budabackend#2730). ZEROACC's face index counts 16-row faces from the start of the
    //    Dest section the math offset selects, so the offset goes back to the section base first.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(section_base);
    const std::uint32_t first_face = dest_row / P.face_r_dim;
    for (std::uint32_t face = 0; face < P.tiles * P.num_faces; face++)
    {
        hal::dst::run(hal::dst::Zero {
            .scope = hal::dst::ZeroScope::Face,
            .index = first_face + face,
            .flags = hal::dst::ZeroFlagAction::ClearFlags,
            .width = hal::dst::DestWidth::Bits32});
    }
}

/// Every tile of the call into Dest from dest_row (relative to the current Dest section).
template <copy_params P>
inline __attribute__((always_inline)) void copy_execute(std::uint32_t dest_row)
{
    if constexpr (P.unpack_to_dest)
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
