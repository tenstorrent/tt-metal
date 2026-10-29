// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: the Dest ring, a dataflow buffer in Dest (tt-llk-lab llk-tensor, report 2, "double banking
// Dest"). Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/).
//
// Dest is split into Sections sections of DEST_REGISTER_FULL_SIZE / Sections rows each: 2 (math writes one while the
// pack thread reads the other, today's SyncHalf) or 1 (SyncFull). A Dest tensor's address is the absolute Dest row it
// starts at, so a section's tensor starts at section * rows; ops add their tile offsets to that row.
//
// The ring is write after read by count: it keeps math from writing a section the pack thread still reads. Read after
// write (math -> pack) is the owner hand-off, inside the op that reads (sync.h), not the ring's.
//   ring_reserve<Math, P>()    math: on the first call only, set dest_ring_sync up; then wait until a section is free
//                              and take it (SEMWAIT while the count is Sections, SEMPOST)
//   ring_release<Pack, P>()    pack: wait for the packer to finish, zero the section, give it back (SEMGET)
//   ring_section_row<P>()      every thread: the first row of this thread's current section
//   ring_next<P>()             every thread: on to the next section; called once per section, by the release
//
// How the threads agree on the section. There is no shared section register. Each TRISC keeps its own copy of the
// index (ring_section, below; every TRISC is a separate program with its own .bss). Every thread runs the same kernel
// body, so every thread reserves and releases the same sections in the same order, and the n-th reserve is section
// n % Sections on every thread. The unpack thread keeps the index too, so an op that unpacks to Dest gets the same
// row; ring_reserve does not make the unpack thread wait, so such an op waits for math itself (copy.h).
//
// No init call. dest_ring_sync is set up inside math's first ring_reserve: value 0, maximum Sections. The pack thread
// touches it only in ring_release, after math has posted it. The first call waits for the value to be 0 first: a
// kernel that ended with a section reserved and not released would leave it above 0.
//
// Dest contents. ring_release zeroes the section after the packer has read it, as today's
// _llk_pack_dest_section_done_ does, so every reserve returns a zeroed section and ops that accumulate (matmul) start
// from 0. The very first sections rely on Dest being zero when the kernel starts (the harness zeroes it in
// device_setup; legacy LLKs rely on the same).
//
// Not HAL: DEST_REGISTER_FULL_SIZE (tensix_types.h) for the section size.

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

/// Rows in one half of Dest: the section size with two sections, and the step between halves for ZEROACC.
inline constexpr std::uint32_t dest_section_rows = DEST_REGISTER_HALF_SIZE;

/// A 32-bit datum takes two 16-bit Dest places (get_dest_max_tiles in ckernel.h).
inline constexpr std::uint32_t dest_places_per_32bit_datum = 2;

/// Parameters of the ring calls.
struct ring_params
{
    std::uint32_t sections;     ///< sections in the ring: 1 or 2
    std::uint32_t section_rows; ///< rows from one section's start to the next
    std::uint32_t rows;         ///< Dest rows the ring's tensor spans from its first row
    bool dest_fp32;             ///< 32-bit Dest mode: zeroing clears 32-bit rows, and a section holds half the rows
};

/// Rows a Dest tensor spans: its offset plus its outermost dim times that dim's stride, in 16-datum rows.
constexpr std::uint32_t dest_rows(const TensorSpec<tensor_rank>& t)
{
    return (t.shape.offset + t.shape.sizes[TileY] * t.shape.strides[TileY]) / dest_row_datums;
}

/// Rows a tensor may use in one section of p's ring: all of it with 16-bit datums, half of it with 32-bit ones.
constexpr std::uint32_t ring_capacity_rows(std::uint32_t section_rows, bool dest_fp32)
{
    return dest_fp32 ? section_rows / dest_places_per_32bit_datum : section_rows;
}

/// The ring parameters for Sections sections of tensors with spec t, in the kernel's Dest mode.
constexpr ring_params ring_params_from(const TensorSpec<tensor_rank>& t, std::uint32_t sections, const KernelSettings& s)
{
    return {sections, DEST_REGISTER_FULL_SIZE / sections, dest_rows(t), s.dest_fp32};
}

/// The section this thread is on. One copy per TRISC, in .bss (zero at kernel start); see the file comment.
inline std::uint32_t ring_section = 0;

/// Math thread: whether dest_ring_sync has been set up in this kernel launch. In .bss, so false at kernel start.
inline bool ring_ready = false;

/// Every thread: the first Dest row of the current section.
template <ring_params P>
inline __attribute__((always_inline)) std::uint32_t ring_section_row()
{
    return ring_section * P.section_rows;
}

/// Every thread: move to the next section. Called once per section, after the section is released.
template <ring_params P>
inline __attribute__((always_inline)) void ring_next()
{
    ring_section = (ring_section + 1) % P.sections;
}

/// Math thread: wait until a section is free, then take it. Sets dest_ring_sync up on the first call.
template <Thread On, ring_params P>
inline __attribute__((always_inline)) void ring_reserve()
{
    static_assert(On == Thread::Math, "only the math thread waits for a free Dest section");
    static_assert(P.sections == 1 || P.sections == 2, "a Dest ring has 1 or 2 sections");
    static_assert(P.rows <= ring_capacity_rows(P.section_rows, P.dest_fp32), "the Dest tensor does not fit one section of the ring: use fewer tiles");
    if constexpr (runs_on<On>)
    {
        if (!ring_ready)
        {
            while (hal::sync::semaphore::read<dest_ring_sync>() > 0)
            {
            }
            hal::sync::semaphore::init<0 /* value */, P.sections /* maximum */, dest_ring_sync>();
            ring_ready = true;
        }
        // Math, SFPU and sync instructions wait here; the next op's Dest writes come after.
        hal::sync::wait::semaphore<StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync, SemaphoreCondition::WhileMaximum, dest_ring_sync>();
        hal::sync::semaphore::post<dest_ring_sync>();
    }
}

/// Pack thread: once the packer has read the section, zero it and give it back to math.
template <Thread On, ring_params P>
inline __attribute__((always_inline)) void ring_release()
{
    static_assert(On == Thread::Pack, "only the pack thread frees a Dest section");
    if constexpr (runs_on<On>)
    {
        // ZEROACC is a matrix-unit instruction: hold it until the packer is idle. The SEMGET behind it waits too.
        hal::sync::wait::stall<StallTarget::Math, StallCondition::PackerIdle>();
        constexpr hal::dst::DestWidth width = P.dest_fp32 ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
        if constexpr (P.sections == 1)
        {
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .index = 0, .width = width});
        }
        else
        {
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = ring_section, .width = width});
        }
        hal::sync::semaphore::get<dest_ring_sync>();
    }
}

} // namespace llk2::llk_lib
