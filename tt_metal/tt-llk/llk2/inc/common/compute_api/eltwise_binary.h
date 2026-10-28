// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: what add (compute_api/add.h) and sub (compute_api/sub.h) share. Single thread, arch-independent.
// Every TRISC runs this same call; llk-lib's per-thread calls execute only on their own thread.
//
// eltwise_binary<Api, Mods>(a, b, acc) resolves the call as a chain of one (resolver/resolver.h): accepts is
// checked at compile time, then the Plan drives llk-lib's unpack (TRISC0) and eltwise_binary (TRISC1). Unpack to
// math is Src dvalid, inside the op. add and sub differ only in Api (add_op, sub_op: resolver/eltwise_binary.h).
//
// acc is a Dest tensor the caller holds (from the Dest sync API); the op overwrites it in place and returns nothing.
// Which thread may touch acc when is the sync API's job, not the op's.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/eltwise_binary.h"
#include "llk_lib/unpack.h"
#include "resolver/eltwise_binary.h"

namespace llk2
{

/// Runs one resolved eltwise binary op: checks acc against the Resolver's result, then unpack and math.
template <const auto& Api, auto Mods, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void eltwise_binary(const Tensor<A>& a, const Tensor<B>& b, const Tensor<D>& acc)
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

} // namespace llk2
