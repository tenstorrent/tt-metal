// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: the Dest ring, a dataflow buffer in Dest (tt-llk-lab llk-tensor, report 2, "double banking
// Dest"). Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/).
//
// Dest is split into Sections sections of DEST_REGISTER_FULL_SIZE / Sections rows each: 2 (math writes one while the
// pack thread reads the other, 1.0's SyncHalf) or 1 (SyncFull). A Dest tensor's address is the absolute Dest row it
// starts at, so a section's tensor starts at section * rows; ops add their tile offsets to that row.
//
// One ring per kernel, many views. The ring is one piece of hardware state: dest_ring_sync and each thread's two
// counters below. A view (ring_params) is the ring's geometry (sections, section_rows, dest_fp32: the same for every
// view of a kernel; the Compute API takes them from kernel_settings.h) plus the rows of the tensor it puts in a
// section. So a kernel may reserve a [1 x 8]-tile tensor, then a [1 x 4] one, then a [1 x 1] one, each in the next
// section, as long as each fits a section.
//
// The ring is write after read by count: it keeps math from writing a section the pack thread still reads. Read after
// write (math -> pack) is the owner hand-off, inside the op that reads (sync.h), not the ring's.
//   ring_reserve<Math, P>()    math: on the first call only, set dest_ring_sync up; then wait until a section is free
//                              and take it (SEMWAIT while the count is Sections, SEMPOST)
//   ring_count_reserve<P>()    every thread: count a reserve and return the first row of the section it takes. The
//                              Compute API calls it in front of ring_reserve, so its LLK_ASSERT catches a reserve
//                              that would wait forever
//   ring_release<Pack, P>()    pack: wait for the packer to finish, zero the oldest reserved section, give it back
//                              (SEMGET)
//   ring_count_release()       every thread: count a release; called once per release, after ring_release
//
// How the threads agree on the section. There is no shared section register. Each TRISC counts its own reserves and
// releases (ring_reserves, ring_releases, below; every TRISC is a separate program with its own .bss). Every thread
// runs the same kernel body, so every thread makes the same reserves and releases in the same order: the n-th reserve
// takes section n % Sections and the n-th release frees it, on every thread. Up to Sections tensors may be reserved
// and not yet released at once; releases come in reserve order. The unpack thread counts too, so an op that unpacks to
// Dest gets the same row; ring_reserve does not make the unpack thread wait, so such an op waits for math itself
// (copy.h).
//
// Misuse that cannot be caught at compile time is caught by LLK_ASSERT (ENABLE_LLK_ASSERT builds):
//   - a reserve while every section is reserved and none released: math would wait forever for a release that comes
//     later in the same kernel body, on every thread
//   - a release without a reserve, or a release out of reserve order (compute_api/dest.h checks the section)
//
// No init call. dest_ring_sync is set up inside math's first ring_reserve, value 0, maximum Sections
// (semaphore_setup_once, semaphores.h). The pack thread touches it only in ring_release, after math has posted it.
// Every view of a kernel has the same Sections, so the first reserve sets the right maximum whichever view it is.
//
// Dest contents. ring_release zeroes the section after the packer has read it, as 1.0's
// _llk_pack_dest_section_done_ does, so every reserve returns a zeroed section and ops that accumulate (matmul) start
// from 0. The very first sections rely on Dest being zero when the kernel starts (the harness zeroes it in
// device_setup; legacy LLKs rely on the same).
//
// Not HAL: DEST_REGISTER_FULL_SIZE (tensix_types.h) for the section size, LLK_ASSERT (llk_assert.h, through hal/dst.h).

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal/sync.h"
#include "kernel_settings.h"
#include "llk_lib/semaphores.h"
#include "llk_lib/thread.h"
#include "tensix_types.h" // DEST_REGISTER_FULL_SIZE
#include "tensor/tensor.h"

namespace llk2::llk_lib
{

using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// A 32-bit datum takes two 16-bit Dest places (get_dest_max_tiles in ckernel.h).
inline constexpr std::uint32_t dest_places_per_32bit_datum = 2;

/// Parameters of the ring calls: a view of the kernel's one ring. Every view of a kernel has the same sections,
/// section_rows and dest_fp32; only rows differs.
struct ring_params
{
    std::uint32_t sections;     ///< sections in the ring: 1 or 2
    std::uint32_t section_rows; ///< rows from one section's start to the next
    std::uint32_t rows;         ///< Dest rows this view's tensor spans from its first row
    bool dest_fp32;             ///< 32-bit Dest mode: zeroing clears 32-bit rows, and a section holds half the rows
};

/// Rows a Dest tensor spans: its offset plus its outermost dim times that dim's stride, in 16-datum rows.
constexpr std::uint32_t dest_rows(const Spec& t)
{
    return (t.shape.offset + t.shape.sizes[TileY] * t.shape.strides[TileY]) / dest_row_datums;
}

/// Rows a tensor may use in a section of section_rows rows: all of them with 16-bit datums, half with 32-bit ones.
constexpr std::uint32_t ring_capacity_rows(std::uint32_t section_rows, bool dest_fp32)
{
    return dest_fp32 ? section_rows / dest_places_per_32bit_datum : section_rows;
}

/// Rows from one section's start to the next in a ring of `sections` sections: Dest split evenly.
constexpr std::uint32_t ring_section_rows(std::uint32_t sections)
{
    return DEST_REGISTER_FULL_SIZE / sections;
}

/// The ring parameters for Sections sections of tensors with spec t, in the kernel's Dest mode.
constexpr ring_params ring_params_from(const Spec& t, std::uint32_t sections, const KernelSettings& s)
{
    return {.sections = sections, .section_rows = ring_section_rows(sections), .rows = dest_rows(t), .dest_fp32 = s.dest_fp32};
}

/// Reserves this thread has made in this kernel launch: the n-th takes section n % sections. One copy per TRISC, in
/// .bss (zero at kernel start); see the file comment.
inline std::uint32_t ring_reserves = 0;

/// Releases this thread has made in this kernel launch: the n-th frees section n % sections. As ring_reserves.
inline std::uint32_t ring_releases = 0;

/// The first Dest row of section `section` of P's ring.
template <ring_params P>
constexpr std::uint32_t ring_row(std::uint32_t section)
{
    return section * P.section_rows;
}

/// Every thread: the section the oldest reserve still not released holds, the one the next release frees.
template <ring_params P>
inline __attribute__((always_inline)) std::uint32_t ring_oldest()
{
    return ring_releases % P.sections;
}

/// Every thread: count a reserve, and return the first Dest row of the section it takes (the next one in ring order).
/// Called in front of ring_reserve: the assert then fires before math would wait forever.
template <ring_params P>
inline __attribute__((always_inline)) std::uint32_t ring_count_reserve()
{
    LLK_ASSERT(ring_reserves - ring_releases < P.sections, "Dest ring: every section is reserved; release one first or math waits forever");
    const std::uint32_t section = ring_reserves % P.sections;
    ring_reserves++;
    return ring_row<P>(section);
}

/// Every thread: count a release. Called once per section, after the section is released.
inline __attribute__((always_inline)) void ring_count_release()
{
    LLK_ASSERT(ring_releases != ring_reserves, "Dest ring: a release without a reserve");
    ring_releases++;
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
        semaphore_setup_once<dest_ring_sync, P.sections>();
        // Math, SFPU and sync instructions wait here; the next op's Dest writes come after.
        hal::sync::wait::semaphore<StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync, SemaphoreCondition::WhileMaximum, dest_ring_sync>();
        hal::sync::semaphore::post<dest_ring_sync>();
    }
}

/// Pack thread: once the packer has read the oldest reserved section, zero it and give it back to math.
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
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = ring_oldest<P>(), .width = width});
        }
        hal::sync::semaphore::get<dest_ring_sync>();
    }
}

} // namespace llk2::llk_lib
