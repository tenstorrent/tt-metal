// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: Dest tensors. Every TRISC runs these calls, and each does its own thread's part (llk_lib/dest.h has
// the hardware side and how the threads agree on the section).
//
// A kernel gets its Dest tensors from a Dest ring. (A test of one op may also name a Dest tensor at a fixed row,
// below.)
//
//   A Dest ring: a dataflow buffer in Dest (tt-llk-lab llk-tensor, report 2, "double banking Dest"). Write after read
//   by count: math waits only when every section is in use, so with two sections math works on the next block while
//   the pack thread still reads the last one.
//
//     using AccRing  = llk2::DestRing<acc_spec>;        // a view of the kernel's Dest ring: each section as acc_spec
//     const auto acc = llk2::reserve_back<AccRing>();   // math: wait for a free section. Every thread: a Dest tensor
//                                                       // at that section's first row, zero, the FPU's
//     const auto y   = llk2::matmul(x, w, acc);         // ops write it and return it (compute_api/sync.h)
//     const auto out = llk2::pack(y, slot);             // the math -> pack hand-off is inside pack
//     llk2::release(y);                                 // pack: once the packer has read it, zero the section and
//                                                       // give it back. Every thread: on to the next section
//
//   For tests, a Dest tensor at a fixed row: llk2::dst_tensor<Rows, Cols, Format>(row) (tensor/tensor.h). Nobody's,
//   so the first op waits for no one. Write after read by identity: complete<Thread::Math>(out) after its last pack, before
//   math writes it again (compute_api/sync.h). Nothing zeroes it.
//
// One ring, many specs. A kernel has one Dest ring: LLK2_DEST_SECTIONS sections (kernel_settings.h; 2, SyncHalf, by
// default), one semaphore, one pair of counters per thread. Every DestRing<spec> is a typed view of it, so a kernel
// mixes Dest tensors of different shapes freely; each reserve_back takes the ring's next section, whichever view it
// goes through:
//
//     using ScoreRing = llk2::DestRing<qk_spec>;        // [1 x 8] tiles: a whole 16-bit section
//     using OutRing   = llk2::DestRing<out_spec>;       // [1 x 4] tiles
//     const auto s = llk2::reserve_back<ScoreRing>();   // section 0
//     const auto o = llk2::reserve_back<OutRing>();     // section 1: both held at once
//     ...  release(s); release(o);                      // in reserve order
//
// Rules:
//   - Every thread must reserve and release every section, in the same order: the threads stay in step only because
//     they make the same calls.
//   - At most LLK2_DEST_SECTIONS tensors reserved and not released at once: one more and math waits forever for a
//     release later in the same kernel body. Release in reserve order. Both are LLK_ASSERTs (ENABLE_LLK_ASSERT).
//   - Release once per section, on any tensor that shares the section (an op's result shares its output's release).
//   - A view's tensor must fit one section: with two sections, 512 rows (8 tiles of 32x32) with a 16-bit Dest, 256
//     rows (4 tiles) with a 32-bit Dest; twice that with one. Its format must match the kernel's Dest mode
//     (kernel_settings.h). Both are checked at compile time.
//   - The sections are the kernel's (LLK2_DEST_SECTIONS), not a view's: a DestRing names only its spec, so two views
//     cannot disagree on the ring's geometry (or the semaphore's maximum).
//   - Nothing to initialize, nothing to undo after the last release.

#pragma once

#include <cstdint>

#include "kernel_settings.h"
#include "llk_lib/dest.h"
#include "tensor/tensor.h"

namespace llk2
{

/// A view of the kernel's Dest ring whose sections each hold one tensor with spec S. A type: name it, then
/// reserve_back from it.
template <const Spec& S>
struct DestRing
{
    /// each section's tensor
    static constexpr const Spec& spec = S;
    /// llk-lib's view
    static constexpr llk_lib::ring_params params = llk_lib::ring_params_from(S, kernel_dest_sections, kernel_settings);
};

/// 32x32 tiles one section of the kernel's Dest ring holds: with two sections 8 in a 16-bit Dest and 4 in a 32-bit
/// one, twice that with one section. What a kernel sizes its blocks by when a tensor is too large for a section.
inline constexpr std::uint32_t dest_section_tiles =
    llk_lib::ring_capacity_rows(llk_lib::ring_section_rows(kernel_dest_sections), kernel_settings.dest_fp32) * dest_row_datums / default_tile_datums;

/// The release action of a Dest ring's tensor: pack gives the section back, every thread moves on.
template <class Ring>
struct DestRingRelease
{
    std::uint32_t row = 0; ///< the first row of the tensor's section: checks that releases come in reserve order

    /// Frees the section (pack: once the packer has read it, zeroed) and moves every thread to the next.
    void operator()() const
    {
        LLK_ASSERT(row == llk_lib::ring_row<Ring::params>(llk_lib::ring_oldest<Ring::params>()), "Dest ring: release in reserve order");
        llk_lib::ring_release<Thread::Pack, Ring::params>();
        llk_lib::ring_count_release();
    }
};

/// Takes the kernel's Dest ring's next section, viewed through Ring. Math waits until it is free. Returns, on every
/// thread, a Dest tensor with Ring's spec at the section's first row; the section is zero and the FPU's (math holds
/// it). release(t) gives it back.
template <class Ring>
inline __attribute__((always_inline)) auto reserve_back()
{
    static_assert(Ring::spec.storage == Storage::Dest, "a Dest ring holds Dest tensors: use a dst_spec");
    static_assert(
        fits_dest_mode(Ring::spec.format, kernel_settings.dest_fp32),
        "the Dest tensor's format must be one of the kernel's Dest mode (LLK2_DEST_FP32): Float32 or Int32 with a 32-bit Dest, else Float16_b, Float16 or "
        "UInt16");
    const std::uint32_t row = llk_lib::ring_count_reserve<Ring::params>(); // first: its LLK_ASSERT catches a reserve that would hang
    llk_lib::ring_reserve<Thread::Math, Ring::params>();
    return Tensor<owned_by<Ring::spec, Thread::Math, Exu::Fpu>, DestRingRelease<Ring>>(row, DestRingRelease<Ring> {row});
}

} // namespace llk2
