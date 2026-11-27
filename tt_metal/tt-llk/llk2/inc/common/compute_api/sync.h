// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: synchronization through owners (tt-llk-lab llk-tensor, report 2 section 7, report 4 section 5).
// Every TRISC runs these calls, and each does its own thread's part (compute_api/modifiers.h).
//
// Read after write needs no call. Every op returns what it wrote as a Tensor owned by its own (thread, unit), and the
// next op that touches it hands it over itself (llk_lib/sync.h):
//
//   const auto y = llk2::matmul(x, w, acc);                          // y: acc's rows, the FPU's (math)
//   const auto z = llk2::vec_silu<llk2::vec_modifiers {.thread = llk2::Thread::Pack}>(y);
//                                                                    // math drains the FPU and posts, the pack
//                                                                    // thread waits; z: the SFPU's (pack)
//   const auto out = llk2::pack(z, slot);                            // the same thread, another unit: a STALLWAIT
//
//   hand-off           read after write                         write after read
//   unpack -> math     Src valid, inside an op                  inside an op
//   FPU -> SFPU        STALLWAIT on the math thread             -
//   math -> pack       SEMPOST / SEMWAIT on math_pack_sync      complete<Math>(out), or a Dest ring (compute_api/dest.h)
//   L1 across kernels  the dataflow buffers (compute_api/dataflow.h): never automatic
//   a scalar           read_value: unpack -> math and pack, through the thread mailboxes (below)
//
// Write after read is explicit:
//   complete<T>(t)           thread T waits until t is written, so (units working in order) t's writer is done
//                            reading what it read. After pack, complete<Thread::Math>(out) frees the Dest tensor pack
//                            read, for a Dest tensor at a fixed row. Leaving it out still compiles: the next write
//                            races the packer. A Dest ring does this by count instead (a missing release hangs)
//   migrate<E, From, To>()   hands unit E (the SFPU) from thread From to thread To, where a kernel moves its SFPU
//                            calls from one thread to the other. Not needed for a kernel whose SFPU calls stay on one
//                            thread, nor for the first SFPU call of a kernel
//
// A scalar on every thread. A value the kernel's control flow needs from a buffer (a chunk's start index, a range of
// blocks) has to be the same on all three threads, or they stop making the same calls:
//   read_value<T>(t, index)  word `index` of t, a tensor at a buffer's front (compute_api/dataflow.h), on every thread.
//                            T is an unsigned integer of 1, 2 or 4 bytes (uint32_t by default); index counts Ts from
//                            t's first byte, inside its first tile (tile(t, i) names another). 1.0's read_tile_value.
//
//   const auto range         = llk2::wait_front<range_spec>(k_range);   // one tile of words from the reader
//   const std::uint32_t k_lo = llk2::read_value(range, 0);
//   const std::uint32_t k_hi = llk2::read_value(range, 1);
//   llk2::release(range);
//
// Who syncs, and when the value is valid (llk_lib/sync.h): the unpack thread loads the word when it reaches the call
// (wait_front has returned, so the page is written) and sends it to math and pack through its mailboxes; each of them
// waits at its own read_value until the value arrives. So on every thread the value is valid when the call returns.
// It is a hand-off of a value between the RISC-V cores, not of a tensor between units: it waits for no unpacker, FPU
// or packer, and t's owner does not change. Flow control is the hardware's: a thread's mailboxes hold four values in
// all, and a write past that waits in the memory subsystem and then stalls the writing core until a reader pops
// (tt-isa-documentation, BabyRISCV/Mailboxes.md), so unpack running ahead of math and pack loses nothing and only
// waits. The values arrive in the order sent, and every thread makes its read_value calls in the same order. t must
// be the unpackers' (from wait_front or a peek) and not yet popped: a tensor this kernel packed has no
// pack -> unpack hand-off (not ported). The rule is read_value's accepts (resolver/read_value.h).

#pragma once

#include <cstdint>
#include <type_traits>

#include "llk_assert.h"
#include "llk_lib/sync.h"
#include "resolver/read_value.h"
#include "tensor/tensor.h"

namespace llk2
{

/// Thread T waits until t is written: t's writer is done, its inputs read. Write after read, by identity.
template <Thread T, const auto& S, class R>
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

/// Word `index` of t (a T, counted in Ts from t's first byte, inside its first tile), the same value on every thread:
/// unpack loads it from L1 and sends it to math and pack, which wait for it here. t is a tensor at a buffer's front.
template <class T = std::uint32_t, const auto& S, class R>
inline __attribute__((always_inline)) T read_value(const Tensor<S, R>& t, std::uint32_t index)
{
    constexpr Accepts ok = read_value_accepts<T>(S);
    static_assert(ok.storage, "read_value reads an L1 tensor at a buffer's front (from wait_front or a peek). Not ported: a tensor this kernel wrote");
    static_assert(ok.formats, "read_value<T>: T is an unsigned integer of 1, 2 or 4 bytes");
    LLK_ASSERT((index + 1) * sizeof(T) <= tile_bytes(S), "read_value: the word is outside the tensor's first tile");
    return llk_lib::read_value<T>(t.address, index);
}

} // namespace llk2
