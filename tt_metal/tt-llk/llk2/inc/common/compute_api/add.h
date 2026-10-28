// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: add on the FPU. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's
// per-thread calls execute only on their own thread. a's and b's L1 addresses are read only on the unpack thread;
// they need to be valid only there (tensor/tensor.h).
//
//   add(a, b, acc);                                              // acc = a + b; a, b in L1, acc in Dest, same shape
//   add<add_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // acc = a + bias's row 0, repeated down every row
//
// With Broadcast::Row, b is one row of tiles (a bias, [32, N] for a's [M, N]); row 0 of tile column c is added to
// every row of a's tile column c. This replaces today's add_tiles / add_tiles_bcast_rows and their inits.
//
// sub (compute_api/sub.h) is the same op with ELWSUB; both go through eltwise_binary (compute_api/eltwise_binary.h).
//
// acc is a Dest tensor the caller holds (from the Dest sync API); the op overwrites it in place and returns nothing.
// Which thread may touch acc when is the sync API's job, not the op's.
//
// The call resolves itself as a chain of one (resolver/resolver.h); compute_api/eltwise_binary.h runs it.

#pragma once

#include "compute_api/eltwise_binary.h"
#include "compute_api/modifiers.h"

namespace llk2
{

/// acc = a + b (b broadcast as Mods.bcast says).
template <add_modifiers Mods = add_modifiers {}, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void add(const Tensor<A>& a, const Tensor<B>& b, const Tensor<D>& acc)
{
    eltwise_binary<add_op, Mods>(a, b, acc);
}

} // namespace llk2
