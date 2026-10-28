// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dest sync API. Single thread, arch-independent: every TRISC runs the same calls, and each call does its own
// thread's part (llk_lib/dest.h has the hardware side and how the threads agree on the section).
//
// One Dest section per block of work, in this order:
//
//   inline constexpr auto& acc_spec = llk2::dst_spec<M, N, llk2::kernel_dest_format>;  // [M, N] in Dest
//
//   const auto acc = llk2::dest_acquire<acc_spec>(); // math: wait for a free section. Every thread: a Dest tensor at
//                                                    // that section's first row (the section is zero)
//   llk2::matmul(x, w, acc);                         // ops write acc (matmul, add, copy, vec_*, ...)
//   llk2::dest_commit(acc);                          // math: done writing, hand the section to pack.
//                                                    // pack: wait until math has
//   llk2::pack(acc, out);                            // pack reads the section (compute_api/pack.h)
//   llk2::dest_release(acc);                         // pack: done reading; zero the section and give it back to math.
//                                                    // every thread: move to the other section
//
// Loop over that block as often as needed: math works on the next section while pack still reads the last one.
// Every thread must make all four calls for every section, in this order, even calls that do nothing on it; the
// threads stay in step only because they make the same calls (llk_lib/dest.h).
//
// Rules for ops (and their authors):
//   - acc.address is the absolute Dest row the tensor starts at (0 or the second section's first row). Program Dest
//     addresses from it (math: DestTargetRegCfgMath::Offset = acc.address + tile rows; pack: the packer's Dest offset)
//     and do not add a section base of your own.
//   - Write Dest only between dest_acquire and dest_commit; read it (pack) only between dest_commit and dest_release.
//   - The tensor must fit one section: 512 rows (8 tiles of 32x32) with a 16-bit Dest, 256 rows (4 tiles) with a
//     32-bit Dest. dest_acquire rejects bigger ones at compile time.
//   - Its format must match the kernel's Dest mode (kernel_settings.h): 32-bit formats with LLK2_DEST_FP32, else
//     16-bit or narrower.
//   - Nothing to initialize: no init call before the first dest_acquire, and nothing to undo after the last
//     dest_release.
//   - On the unpack thread dest_acquire does not wait. An op that unpacks straight to Dest must wait for math to own
//     the section itself.

#pragma once

#include "kernel_settings.h"
#include "llk_lib/dest.h"
#include "tensor/tensor.h"

namespace llk2
{

/// Takes the next Dest section. Math waits until it is free (the packer has released it). Returns, on every thread,
/// a Dest tensor with spec S starting at the section's first row; the section is zero.
template <const TensorSpec<tensor_rank>& S>
inline __attribute__((always_inline)) Tensor<S> dest_acquire()
{
    static_assert(S.storage == Storage::Dest, "dest_acquire returns a Dest tensor: use a dst_spec");
    static_assert(
        kernel_settings.dest_fp32 ? datum_bits(S.format) == 32 : datum_bits(S.format) <= 16 && datum_bits(S.format) > 0,
        "the Dest tensor's format must match the kernel's Dest mode (LLK2_DEST_FP32)");
    constexpr llk_lib::dest_params p = llk_lib::dest_params_from(S, kernel_settings);
    llk_lib::dest_acquire<Thread::Math, p>();
    return Tensor<S>(llk_lib::dest_section_row());
}

/// Ends the writes to acc's section. Math hands it to the packer once math and the SFPU are done; pack waits for that.
template <const TensorSpec<tensor_rank>& S>
inline __attribute__((always_inline)) void dest_commit(const Tensor<S>&)
{
    static_assert(S.storage == Storage::Dest, "dest_commit takes the tensor dest_acquire returned");
    constexpr llk_lib::dest_params p = llk_lib::dest_params_from(S, kernel_settings);
    llk_lib::dest_commit<Thread::Math, p>();
    llk_lib::dest_commit<Thread::Pack, p>();
}

/// Ends the reads of acc's section. Pack zeroes it and hands it back to math once the packer is done; every thread
/// moves to the other section. acc must not be used after this.
template <const TensorSpec<tensor_rank>& S>
inline __attribute__((always_inline)) void dest_release(const Tensor<S>&)
{
    static_assert(S.storage == Storage::Dest, "dest_release takes the tensor dest_acquire returned");
    constexpr llk_lib::dest_params p = llk_lib::dest_params_from(S, kernel_settings);
    llk_lib::dest_release<Thread::Pack, p>();
    llk_lib::dest_next_section();
}

} // namespace llk2
