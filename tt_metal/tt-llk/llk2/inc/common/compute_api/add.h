// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: add on the FPU. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's
// per-thread calls execute only on their own thread. a's and b's L1 addresses are read only on the unpack thread;
// they need to be valid only there (tensor/tensor.h).
//
//   y = add(a, b, acc);                                              // y = a + b; a, b in L1, acc in Dest, same shape
//   y = add<add_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // y = a + bias's row 0, repeated down every row
//
// With Broadcast::Row, b is one row of tiles (a bias, [32, N] for a's [M, N]); row 0 of tile column c is added to
// every row of a's tile column c. This replaces today's add_tiles / add_tiles_bcast_rows and their inits.
//
// sub (compute_api/sub.h) is the same op with ELWSUB; both go through eltwise_binary (compute_api/eltwise_binary.h).
//
// acc is a Dest tensor (compute_api/dest.h); the op overwrites it in place and returns it as y: the same rows and
// release, now the FPU's. It takes a and b over for the unpackers and acc for the FPU first (compute_api/sync.h).
//
// The call resolves itself as a chain of one (resolver/resolver.h); compute_api/eltwise_binary.h runs it.

#pragma once

#include "compute_api/eltwise_binary.h"
#include "compute_api/modifiers.h"

namespace llk2
{

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

} // namespace llk2
