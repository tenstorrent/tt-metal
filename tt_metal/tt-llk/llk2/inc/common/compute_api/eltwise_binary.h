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
// acc is a Dest tensor (compute_api/dest.h); the op overwrites it in place and returns it, the FPU's. The hand-offs
// to the unpackers and the FPU are taken first (compute_api/sync.h).

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/eltwise_binary.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/eltwise_binary.h"

namespace llk2
{

/// Runs one resolved eltwise binary op: checks acc against the Resolver's result, takes the tensors over, then unpack
/// and math. Returns acc as the FPU's.
template <const auto& Api, auto Mods, const TensorSpec<tensor_rank>& A, class RA, const TensorSpec<tensor_rank>& B, class RB, const TensorSpec<tensor_rank>& D, class RD>
inline __attribute__((always_inline)) auto eltwise_binary(const Tensor<A, RA>& a, const Tensor<B, RB>& b, const Tensor<D, RD>& acc)
{
    static_assert(D.storage == Storage::Dest, "an eltwise binary op writes Dest");
    static_assert(
        accepted<Api, A, B, Mods>,
        "eltwise binary takes 32x32 float tiles in L1 with the same exponent width; b has a's shape, or with Broadcast::Row is one row of a's tile "
        "columns");

    constexpr const Resolved& r = resolved<Api, A, B, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format, "acc must have a's shape and the kernel's Dest format");

    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, A>();
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, B>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, D>();
    llk_lib::unpack<Thread::Unpack, r.plan>(a.address, b.address);
    llk_lib::eltwise_binary<Thread::Math, r.plan>(acc.address);
    return as_owned<Thread::Math, Exu::Fpu>(acc);
}

} // namespace llk2
