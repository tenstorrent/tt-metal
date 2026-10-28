// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: matmul. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's per-thread calls
// execute only on their own thread. in0's and in1's L1 addresses are read only on the unpack thread; they need to
// be valid only there (tensor/tensor.h).
//
//   matmul(in0, in1, acc);                              // acc += in0 * in1; in0: [M, K], in1: [K, N], acc: [M, N] in Dest
//   matmul<matmul_modifiers{...}>(in0, in1, acc);
//
// acc is a Dest tensor the caller holds (from the Dest sync API); matmul adds into it in place and returns nothing.
// Which thread may touch acc when is the sync API's job, not the op's.
//
// The call resolves itself as a chain of one (resolver/resolver.h): accepts is checked at compile time, then the
// Plan drives llk-lib's unpack (TRISC0) and matmul (TRISC1). Unpack to math is Src dvalid, inside the op.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/matmul.h"
#include "llk_lib/unpack.h"
#include "resolver/matmul.h"

namespace llk2
{

/// acc += in0 * in1, every K step. in0 [M, K] and in1 [K, N] in L1, acc [M, N] in Dest.
template <matmul_modifiers Mods = matmul_modifiers {}, const TensorSpec<tensor_rank>& A, const TensorSpec<tensor_rank>& B, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void matmul(const Tensor<A>& in0, const Tensor<B>& in1, const Tensor<D>& acc)
{
    static_assert(D.storage == Storage::Dest, "matmul accumulates into Dest");
    static_assert(
        accepted<matmul_op, A, B, Mods>, "matmul takes 32x32 tiles in L1 with matching K, in formats the unpacker can write to Src, no transpose yet");

    constexpr const Resolved& r = resolved<matmul_op, A, B, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
        "acc must have matmul's output shape [M, N] and the kernel's Dest format");

    llk_lib::unpack<Thread::Unpack, r.plan>(in0.address, in1.address);
    llk_lib::matmul<Thread::Math, r.plan>(acc.address);
}

} // namespace llk2
