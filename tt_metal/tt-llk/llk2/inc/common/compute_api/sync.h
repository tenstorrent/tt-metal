// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: synchronization through owners (tt-llk-lab llk-tensor, report 2 section 7, report 4 section 5).
// Single thread, arch-independent: every TRISC runs these calls, and each does its own thread's part.
//
// Read after write needs no call. Every op returns what it wrote as a Tensor owned by its own (thread, unit), and the
// next op that touches it hands it over itself (llk_lib/sync.h):
//
//   const auto y = llk2::matmul(x, w, acc);                          // y: acc's rows, the FPU's (math)
//   const auto z = llk2::vec_silu<{.thread = Thread::Pack}>(y);      // math drains the FPU and posts, the pack
//                                                                    // thread waits; z: the SFPU's (pack)
//   const auto out = llk2::pack(z, slot);                            // the same thread, another unit: a STALLWAIT
//
//   hand-off           read after write                         write after read
//   unpack -> math     Src valid, inside an op                  inside an op
//   FPU -> SFPU        STALLWAIT on the math thread             -
//   math -> pack       SEMPOST / SEMWAIT on math_pack_sync      complete<Math>(out), or a Dest ring (compute_api/dest.h)
//   L1 across kernels  the dataflow buffers (compute_api/dataflow.h): never automatic
//
// Write after read is explicit:
//   complete<T>(t)           thread T waits until t is written, so (units working in order) t's writer is done
//                            reading what it read. After pack, complete<Thread::Math>(out) frees the Dest tensor pack
//                            read, for a Dest tensor at a fixed row. Leaving it out still compiles: the next write
//                            races the packer. A Dest ring does this by count instead (a missing release hangs)
//   migrate<E, From, To>()   hands unit E (the SFPU) from thread From to thread To, where a kernel moves its SFPU
//                            calls from one thread to the other. Not needed for a kernel whose SFPU calls stay on one
//                            thread, nor for the first SFPU call of a kernel

#pragma once

#include "llk_lib/sync.h"
#include "tensor/tensor.h"

namespace llk2
{

/// Thread T waits until t is written: t's writer is done, its inputs read. Write after read, by identity.
template <Thread T, const TensorSpec<tensor_rank>& S, class R>
inline __attribute__((always_inline)) void complete(const Tensor<S, R>&)
{
    static_assert(T != Thread::None, "complete names the thread that waits");
    llk_lib::acquire<T, Exu::None, S>();
}

/// Hands unit E from thread From to thread To: From waits until E is idle and posts, To waits.
template <Exu E, Thread From, Thread To>
inline __attribute__((always_inline)) void migrate()
{
    llk_lib::migrate<E, From, To>();
}

} // namespace llk2
