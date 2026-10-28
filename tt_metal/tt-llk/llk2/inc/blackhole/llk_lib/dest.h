// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Dest sync on Blackhole: how the math thread hands a Dest section to the pack thread and gets it back.
// Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/).
//
// Dest is split into two sections of dest_section_rows rows (SyncHalf, as today): math writes one while the packer
// reads the other. A Dest tensor's address is the absolute Dest row it starts at, so a section's tensor starts at
// section * dest_section_rows. Ops add their tile offsets to that row and do not add a section base of their own.
//
// The calls, in the order they run in each section (the Compute API, compute_api/dest.h, makes them on every thread):
//   dest_section_row()          every thread: the first row of the current section
//   dest_acquire<Math, P>()     math: on the first call only, set up MATH_PACK; then wait for a free section
//   dest_commit<Math, P>()      math: wait for math and the SFPU to finish writing, then post MATH_PACK (one more
//                               section for the packer)
//   dest_commit<Pack, P>()      pack: hold every later pack-thread instruction while MATH_PACK is 0 (nothing to
//                               pack yet)
//   dest_release<Pack, P>()     pack: wait for the packer to finish, zero the section, then get MATH_PACK (one
//                               section free again)
//   dest_next_section()         every thread: move to the other section
//
// The handshake. Semaphore MATH_PACK counts the sections math has finished and the packer has not yet freed. Its
// maximum is dest_sections (2): math waits while it is at the maximum, the packer waits while it is 0.
//
// How the threads agree on the section. There is no shared section register. Each TRISC keeps its own copy of the
// section index (dest_section, below; every TRISC is a separate program with its own .bss). Every thread runs the
// same kernel body, so every thread makes the same Compute API calls in the same order, and each flips its copy once
// per section, in dest_release. So on every thread the n-th dest_acquire returns section n % 2. MATH_PACK only keeps
// math at most dest_sections sections ahead of pack, so math's section n and pack's section n are the same rows.
// The unpack thread keeps the index too, so an op that unpacks to Dest gets the same row; dest_acquire does not
// make the unpack thread wait, so such an op must wait for math to own the section itself (today's
// math_unpack_to_dest_math_ready handshake).
//
// No init call. Today's _llk_math_pack_sync_init_ (SEMINIT of MATH_PACK) and _llk_pack_dest_init_ (the packer's
// Dest offset GPRs and address counters) have no separate call here:
//   - MATH_PACK is set up inside the math thread's first dest_acquire, before its first wait. The ISA resets a
//     semaphore to Value 0, Max 15 (tt-isa-documentation, BlackholeA0/.../SyncUnit.md); neither the harness
//     firmware (tests/helpers/include/boot.h) nor tt-metal's (trisc.cc) sets MATH_PACK, and Max 15 would let math
//     run 15 sections ahead, so the SEMINIT is needed. It is sound because the packer touches MATH_PACK only after
//     math posts it: the packer's first dest_commit waits while it is 0, which holds before and after the SEMINIT.
//     The first call also waits for the value to be 0 first, as _llk_math_pack_sync_init_ does.
//     Risks: (1) it needs MATH_PACK's value to be 0 when the kernel starts, as today (a kernel that ends with a
//     section committed and not released leaves it at 1, and the next kernel's packer would read that section
//     early); (2) dest_sync_ready lives in .bss, which the harness (do_crt0) and tt-metal (do_crt1 per kernel launch)
//     zero, so every kernel launch sets MATH_PACK up again; a kernel whose .bss is not zeroed would skip it.
//   - The packer's Dest side needs no setup: pack (llk_lib/pack.h) writes the packer's Dest offset from the Dest
//     tensor's address and resets its address counters on every call.
//
// Dest contents. dest_release zeroes the section after the packer has read it, as today's _llk_pack_dest_section_done_
// does, so each dest_acquire returns a zeroed section and ops that accumulate (matmul) can start from 0. The very
// first section relies on Dest being zero when the kernel starts (the harness zeroes it in device_setup; legacy LLKs
// rely on the same).
//
// Only SyncHalf is ported: two sections. Not HAL: DEST_REGISTER_HALF_SIZE (tensix_types.h) for the section size.

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal/sync.h"
#include "kernel_settings.h"
#include "llk_lib/semaphores.h"
#include "llk_lib/thread.h"
#include "tensix_types.h" // DEST_REGISTER_HALF_SIZE, DEST_REGISTER_FULL_SIZE
#include "tensor/tensor.h"

namespace llk2::llk_lib
{

using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

inline constexpr std::uint32_t dest_sections   = 2;                       ///< SyncHalf: math writes one section while pack reads the other
inline constexpr std::uint32_t dest_section_rows = DEST_REGISTER_HALF_SIZE; ///< rows from one section's start to the next, in either Dest mode
static_assert(dest_sections * dest_section_rows == DEST_REGISTER_FULL_SIZE, "the sections split Dest");

/// Rows a Dest tensor may use in one section: all of it with 16-bit datums, half with 32-bit ones (a 32-bit datum
/// takes two 16-bit places; get_dest_max_tiles in ckernel.h).
constexpr std::uint32_t dest_section_capacity_rows(bool dest_fp32)
{
    return dest_fp32 ? dest_section_rows / 2 : dest_section_rows;
}

/// Parameters of the Dest sync calls. The section is the same for every Dest tensor; the tensor only sets how many
/// rows it needs.
struct dest_params
{
    std::uint32_t rows; ///< Dest rows the tensor spans from its first row
    bool dest_fp32;     ///< 32-bit Dest mode: the section zeroing clears 32-bit rows
};

/// Rows a Dest tensor spans: its offset plus its outermost dim times that dim's stride, in 16-datum rows.
constexpr std::uint32_t dest_rows(const TensorSpec<tensor_rank>& t)
{
    return (t.shape.offset + t.shape.sizes[TileY] * t.shape.strides[TileY]) / dest_row_datums;
}

/// The Dest sync parameters for a tensor with spec t, in the kernel's Dest mode.
constexpr dest_params dest_params_from(const TensorSpec<tensor_rank>& t, const KernelSettings& s)
{
    return {dest_rows(t), s.dest_fp32};
}

/// The section this thread is on: 0 or 1. One copy per TRISC, in .bss (zero at kernel start); see the file comment.
inline std::uint32_t dest_section = 0;

/// Math thread: whether MATH_PACK has been set up in this kernel launch. In .bss, so false at kernel start.
inline bool dest_sync_ready = false;

/// Every thread: the first Dest row of the current section.
inline __attribute__((always_inline)) std::uint32_t dest_section_row()
{
    return dest_section * dest_section_rows;
}

/// Every thread: move to the other section. Called once per section, after the section is released.
inline __attribute__((always_inline)) void dest_next_section()
{
    dest_section = (dest_section + 1) % dest_sections;
}

/// Math thread: sets MATH_PACK to value 0, maximum dest_sections, once the packer has freed every section.
inline __attribute__((always_inline)) void dest_sync_setup()
{
    while (hal::sync::semaphore::read<math_pack_sync>() > 0)
    {
    }
    hal::sync::semaphore::init<0 /* value */, dest_sections /* maximum */, math_pack_sync>();
    dest_sync_ready = true;
}

/// Math thread: wait until a section is free (MATH_PACK below its maximum). Sets MATH_PACK up on the first call.
template <Thread On, dest_params P>
inline __attribute__((always_inline)) void dest_acquire()
{
    static_assert(On == Thread::Math, "only the math thread waits for a free Dest section");
    static_assert(P.rows <= dest_section_capacity_rows(P.dest_fp32), "the Dest tensor does not fit one Dest section (SyncHalf); use fewer tiles");
    if constexpr (runs_on<On>)
    {
        if (!dest_sync_ready)
        {
            dest_sync_setup();
        }
        // Math, SFPU and sync instructions wait here; the next op's Dest writes come after.
        hal::sync::wait::semaphore<StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync, SemaphoreCondition::WhileMaximum, math_pack_sync>();
    }
}

/// Math: hand the section to the packer once math and the SFPU are done writing it. Pack: wait until math has.
template <Thread On, dest_params P>
inline __attribute__((always_inline)) void dest_commit()
{
    static_assert(On == Thread::Math || On == Thread::Pack, "Dest commit is a math or pack thread call");
    if constexpr (On == Thread::Math && runs_on<On>)
    {
        hal::sync::wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::SfpuIdle>();
        hal::sync::semaphore::post<math_pack_sync>();
    }
    if constexpr (On == Thread::Pack && runs_on<On>)
    {
        // Every instruction class waits here (block mask All), so any pack-thread work after dest_commit runs only
        // once math has handed the section over: PACR, ADC and THCON (Tdma), mutexes and the release's SEMGET (Sync,
        // see _llk_packer_wait_for_math_done_), config writes (Config: RMWCIB, WRCFG, SETC16), SFPU instructions
        // (Sfpu: activations run from the pack thread) and ZEROACC (Math). The SEMWAIT's block mask names the
        // instruction classes held at the Wait Gate; a class not named passes it while the wait is latched
        // (tt-isa-documentation, BlackholeA0/TensixTile/TensixCoprocessor/SEMWAIT.md, block-mask bits in
        // STALLWAIT.md). RISC-V MMIO writes do not go through the Wait Gate, so pack-thread code after dest_commit
        // must write config through the Tensix config unit, not MMIO, or wait on MATH_PACK itself.
        hal::sync::wait::semaphore<StallTarget::All, SemaphoreCondition::WhileZero, math_pack_sync>();
    }
}

/// Pack thread: once the packer has read the section, zero it and give it back to math.
template <Thread On, dest_params P>
inline __attribute__((always_inline)) void dest_release()
{
    static_assert(On == Thread::Pack, "only the pack thread frees a Dest section");
    if constexpr (runs_on<On>)
    {
        // ZEROACC is a matrix-unit instruction: hold it until the packer is idle. The SEMGET behind it waits too.
        hal::sync::wait::stall<StallTarget::Math, StallCondition::PackerIdle>();
        hal::dst::run(hal::dst::Zero {
            .scope = hal::dst::ZeroScope::Half, .index = dest_section, .width = P.dest_fp32 ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16});
        hal::sync::semaphore::get<math_pack_sync>();
    }
}

} // namespace llk2::llk_lib
