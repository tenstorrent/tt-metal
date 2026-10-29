// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: add and sub on the FPU. Single thread, arch-independent. Every TRISC runs these same calls; llk-lib's
// per-thread calls execute only on their own thread. a's and b's L1 addresses are read only on the unpack thread;
// they need to be valid only there (tensor/tensor.h).
//
//   y = add(a, b, acc);                                              // y = a + b; a, b in L1, acc in Dest, same shape
//   y = add<add_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // y = a + bias's row 0, repeated down every row
//   y = sub(a, b, acc);                                              // y = a - b
//   y = sub<sub_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // y = a - bias's row 0
//
// With Broadcast::Row, b is one row of tiles (a bias, [32, N] for a's [M, N]); row 0 of tile column c is added to (or
// subtracted from) every row of a's tile column c. This replaces today's add_tiles / sub_tiles, their
// *_bcast_rows / BroadcastType::ROW forms and their inits.
//
// acc is a Dest tensor (compute_api/dest.h); the op overwrites it in place and returns it as y: the same rows and
// release, now the FPU's. It takes a and b over for the unpackers and acc for the FPU first (compute_api/sync.h).
//
// Both resolve themselves as a chain of one (resolver/resolver.h) through eltwise_binary below: accepts is checked at
// compile time, then the Plan drives llk-lib's unpack (TRISC0) and eltwise_binary (TRISC1). Unpack to math is Src
// dvalid, inside the op. add and sub differ only in their OpApi (add_op, sub_op: resolver/eltwise_binary.h).

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/eltwise_binary.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/eltwise_binary.h"

namespace llk2
{

/// What add and sub share. Runs one resolved eltwise binary op: checks acc against the Resolver's result, takes the tensors over, then unpack
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

/// a + b (b broadcast as Mods.bcast says), written to acc and returned as the FPU's.
template <
    add_modifiers Mods = add_modifiers {},
    const TensorSpec<tensor_rank>& A,
    class RA,
    const TensorSpec<tensor_rank>& B,
    class RB,
    const TensorSpec<tensor_rank>& D,
    class RD>
inline __attribute__((always_inline)) auto add(const Tensor<A, RA>& a, const Tensor<B, RB>& b, const Tensor<D, RD>& acc)
{
    return eltwise_binary<add_op, Mods>(a, b, acc);
}

/// a - b (b broadcast as Mods.bcast says), written to acc and returned as the FPU's.
template <
    sub_modifiers Mods = sub_modifiers {},
    const TensorSpec<tensor_rank>& A,
    class RA,
    const TensorSpec<tensor_rank>& B,
    class RB,
    const TensorSpec<tensor_rank>& D,
    class RD>
inline __attribute__((always_inline)) auto sub(const Tensor<A, RA>& a, const Tensor<B, RB>& b, const Tensor<D, RD>& acc)
{
    return eltwise_binary<sub_op, Mods>(a, b, acc);
}

} // namespace llk2
