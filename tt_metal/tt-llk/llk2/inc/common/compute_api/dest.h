// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: Dest tensors. Single thread, arch-independent: every TRISC runs these calls, and each does its own
// thread's part (llk_lib/dest.h has the hardware side and how the threads agree on the section).
//
// Two ways to get a Dest tensor for ops to write:
//
//   A Dest ring: a dataflow buffer in Dest (tt-llk-lab llk-tensor, report 2, "double banking Dest"). Write after read
//   by count: math waits only when every section is in use, so with two sections math works on the next block while
//   the pack thread still reads the last one.
//
//     using AccRing  = llk::DestRing<decltype(acc_spec)>;   // 2 sections (SyncHalf); DestRing<decltype(acc_spec), 1>: one
//     const auto acc = llk::reserve_back<AccRing>();   // math: wait for a free section. Every thread: a Dest tensor
//                                                       // at that section's first row, zero, the FPU's
//     const auto y   = llk::matmul(x, w, acc);         // ops write it and return it (compute_api/sync.h)
//     const auto out = llk::pack(y, slot);             // the math -> pack hand-off is inside pack
//     llk::release(y);                                 // pack: once the packer has read it, zero the section and
//                                                       // give it back. Every thread: on to the next section
//
//   A Dest tensor at a fixed row: llk::dst_tensor<Rows, Cols, Format>(row) (tensor/tensor.h). Nobody's, so the
//   first op waits for no one. Write after read by identity: complete<OwnerThread::Compute1>(out) after its last pack, before
//   math writes it again (compute_api/sync.h). Nothing zeroes it.
//
// Rules:
//   - Every thread must reserve and release every section, in the same order: the threads stay in step only because
//     they make the same calls. One ring per kernel: the ring's semaphore and each thread's section index are global.
//   - Release once per section, on any tensor that shares the section (an op's result shares its output's release).
//   - A ring's tensor must fit one section: with two sections, 512 rows (8 tiles of 32x32) with a 16-bit Dest, 256
//     rows (4 tiles) with a 32-bit Dest; twice that with one. Its format must match the kernel's Dest mode
//     (kernel_settings.h). Both are checked at compile time.
//   - Nothing to initialize, nothing to undo after the last release.

#pragma once

#include <cstdint>

#include "kernel_settings.h"
#include "llk_lib/dest.h"
#include "tensor/tensor.h"

namespace llk
{

/// A ring of Sections Dest sections, each holding one tensor of spec type S: DestRing<decltype(acc_spec)>. A type: name
/// it, then reserve_back from it.
template <class S, std::uint32_t Sections = 2>
struct DestRing
{
    using Spec = stdx::remove_cvref_t<S>;                                                                           ///< each section's tensor
    static constexpr llk_lib::ring_params params = llk_lib::ring_params_from(layout_of<S>, Sections, kernel_settings); ///< llk-lib's view
};

/// The release policy of a Dest ring's tensor: pack gives the section back, every thread moves on.
template <class Ring>
struct DestRingRelease
{
    /// Frees the section (pack: once the packer has read it, zeroed) and moves every thread to the next.
    inline __attribute__((always_inline)) void operator()() const
    {
        llk_lib::ring_release<OwnerThread::Compute2, Ring::params>();
        llk_lib::ring_next<Ring::params>();
    }
};

/// Takes Ring's next section. Math waits until it is free. Returns, on every thread, a Dest tensor with Ring's spec at
/// the section's first row; the section is zero and the FPU's (math holds it). release(t) gives it back.
template <class Ring>
inline __attribute__((always_inline)) auto reserve_back()
{
    static_assert(Ring::Spec::storage() == Storage::Dest, "a Dest ring holds Dest tensors: use a dst_spec");
    static_assert(
        kernel_settings.dest_fp32 ? datum_bits(Ring::Spec::format()) == 32 : datum_bits(Ring::Spec::format()) <= 16 && datum_bits(Ring::Spec::format()) > 0,
        "the Dest tensor's format must match the kernel's Dest mode (LLK2_DEST_FP32)");
    llk_lib::ring_reserve<OwnerThread::Compute1, Ring::params>();
    return Tensor<typename Ring::Spec, DestRingRelease<Ring>>(typename Ring::Spec {}, llk_lib::ring_section_row<Ring::params>(), OwnerThread::Compute1, OwnerExu::Fpu);
}

} // namespace llk
