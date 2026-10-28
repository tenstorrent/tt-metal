// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: sub on the FPU. Single thread, arch-independent. The same op as add (compute_api/add.h), with in0 - in1
// instead of in0 + in1:
//
//   sub(a, b, acc);                                              // acc = a - b; a, b in L1, acc in Dest, same shape
//   sub<sub_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // acc = a - bias's row 0, repeated down every row
//
// This replaces today's sub_tiles / sub_tiles_bcast<BroadcastType::ROW> and their inits.

#pragma once

#include "compute_api/add.h" // eltwise_binary

namespace llk2
{

/// acc = a - b (b broadcast as Mods.bcast says).
template <sub_modifiers Mods = sub_modifiers {}, const TensorSpec<tensor_rank>& A, class RA, const TensorSpec<tensor_rank>& B, class RB, const TensorSpec<tensor_rank>& D, class RD>
inline __attribute__((always_inline)) void sub(const Tensor<A, RA>& a, const Tensor<B, RB>& b, const Tensor<D, RD>& acc)
{
    eltwise_binary<sub_op, Mods>(a, b, acc);
}

} // namespace llk2
