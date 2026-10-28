// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: add on the FPU. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's
// per-thread calls execute only on their own thread.
//
//   add(a, b, acc);                                              // acc = a + b; a, b in L1, acc in Dest, same shape
//   add<add_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // acc = a + bias's row 0, repeated down every row
//
// With Broadcast::Row, b is one row of tiles (a bias, [32, N] for a's [M, N]); row 0 of tile column c is added to
// every row of a's tile column c. This replaces today's add_tiles / add_tiles_bcast_rows and their inits.
//
// sub (compute_api/sub.h) is the same op with ELWSUB; both go through eltwise_binary below.
//
// acc is a Dest tensor the caller holds (from the Dest sync API); the op overwrites it in place and returns nothing.
// Which thread may touch acc when is the sync API's job, not the op's.
//
// The call resolves itself as a chain of one (resolver/resolver.h): accepts is checked at compile time, then the
// Plan drives llk-lib's unpack (TRISC0) and eltwise_binary (TRISC1). Unpack to math is Src dvalid, inside the op.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/eltwise_binary.h"
#include "llk_lib/unpack.h"
#include "resolver/eltwise_binary.h"

namespace llk2
{

/// Runs one resolved eltwise binary op: checks acc against the Resolver's result, then unpack and math.
template <const auto& Api, auto Mods, const TensorSpec<tensor_rank>& A, class RA, const TensorSpec<tensor_rank>& B, class RB, const TensorSpec<tensor_rank>& D, class RD>
inline __attribute__((always_inline)) void eltwise_binary(const Tensor<A, RA>& a, const Tensor<B, RB>& b, const Tensor<D, RD>& acc)
{
    static_assert(D.storage == Storage::Dest, "an eltwise binary op writes Dest");
    static_assert(
        accepted<Api, A, B, Mods>,
        "eltwise binary takes 32x32 float tiles in L1 with the same exponent width; b has a's shape, or with Broadcast::Row is one row of a's tile "
        "columns");

    constexpr const Resolved& r = resolved<Api, A, B, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format, "acc must have a's shape and the kernel's Dest format");

    llk_lib::unpack<Thread::Unpack, r.plan>(a.address, b.address);
    llk_lib::eltwise_binary<Thread::Math, r.plan>(acc.address);
}

/// acc = a + b (b broadcast as Mods.bcast says).
template <add_modifiers Mods = add_modifiers {}, const TensorSpec<tensor_rank>& A, class RA, const TensorSpec<tensor_rank>& B, class RB, const TensorSpec<tensor_rank>& D, class RD>
inline __attribute__((always_inline)) void add(const Tensor<A, RA>& a, const Tensor<B, RB>& b, const Tensor<D, RD>& acc)
{
    eltwise_binary<add_op, Mods>(a, b, acc);
}

} // namespace llk2
