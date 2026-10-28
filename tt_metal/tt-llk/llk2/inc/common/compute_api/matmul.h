// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: matmul. Single thread, arch-independent. Every TRISC runs this same call; the LLK API's per-thread
// calls execute only on their own thread.
//
//   const auto y = matmul(x, w, acc);                                     // x: [M, K], w: [K, N], acc: [M, N] in Dest
//   const auto y = matmul<matmul_modifiers{.transpose_in1 = true}>(x, w, acc);
//
// The call resolves itself as a chain of one (resolver/resolver.h) and returns the Dest tensor, now the FPU's.
// Unpack to math is Src dvalid, inside the op. Math to pack is not handled here yet: the caller syncs Dest.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_api/matmul.h"
#include "resolver/matmul.h"

namespace llk2
{

template <matmul_modifiers Mods = matmul_modifiers {}, const TensorSpec<6>& A, class RA, const TensorSpec<6>& B, class RB, const TensorSpec<6>& D, class RD>
inline __attribute__((always_inline)) auto matmul(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& out)
{
    static_assert(D.storage == Storage::Dest, "matmul writes Dest");
    static_assert(free_or(D, Thread::Math, Exu::Fpu), "matmul writes Dest that is nobody's or the FPU's");

    constexpr const Resolved& r = resolved<matmul_op, A, B, Mods>;
    static_assert(equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
                  "the Dest tensor must have matmul's output shape and the kernel's Dest format");

    unpack_matmul<Thread::Unpack, r.plan>(in0.address, in1.address);
    math_matmul<Thread::Math, r.plan>(out.address);
    return Tensor<owned_by<D, Thread::Math, Exu::Fpu>, RD>(out.address);
}

} // namespace llk2
