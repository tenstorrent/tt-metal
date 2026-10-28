// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: matmul. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's per-thread calls
// execute only on their own thread.
//
//   const auto y = matmul(x, w, 0);                    // x: [M, K], w: [K, N]; y: [M, N] in Dest from row 0
//   const auto y = matmul(x, w, acc);                  // or into a Dest tensor the caller made (dst_tensor)
//   const auto y = matmul<matmul_modifiers{...}>(x, w, 0);
//
// Why an output argument: matmul does not allocate Dest. The caller says where in Dest the result goes, either as
// a row (matmul lays the output tensor out from its inputs) or as a Dest tensor it built. y is that same Dest
// region, so it has the same address; what changes is its type: y's spec says the FPU owns it now, which is what
// the next op (pack, an SFPU op) syncs on.
//
// Unpack to math is Src dvalid, inside the op. Math to pack is not handled here yet: the caller syncs Dest.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/matmul.h"
#include "llk_lib/unpack.h"
#include "resolver/matmul.h"

namespace llk2
{

template <matmul_modifiers Mods = matmul_modifiers {}, const TensorSpec<6>& A, class RA, const TensorSpec<6>& B, class RB, const TensorSpec<6>& D, class RD>
inline __attribute__((always_inline)) auto matmul(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& out)
{
    static_assert(D.storage == Storage::Dest, "matmul writes Dest");
    static_assert(free_or(D, Thread::Math, Exu::Fpu), "matmul writes Dest that is nobody's or the FPU's");
    static_assert(accepted<matmul_op, A, B, Mods>, "matmul takes 32x32 tiles in L1 with matching K, in formats the unpacker can write to Src, no transpose yet");

    constexpr const Resolved& r = resolved<matmul_op, A, B, Mods>;
    static_assert(equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
                  "the Dest tensor must have matmul's output shape and the kernel's Dest format");

    llk_lib::unpack<Thread::Unpack, r.plan>(in0.address, in1.address);
    llk_lib::matmul<Thread::Math, r.plan>(out.address);
    return Tensor<owned_by<D, Thread::Math, Exu::Fpu>, RD>(out.address);
}

// The spec matmul gives its output, named so it can be a Tensor's template argument.
template <const TensorSpec<6>& A, const TensorSpec<6>& B, matmul_modifiers Mods>
inline constexpr TensorSpec<6> matmul_result = resolved<matmul_op, A, B, Mods>.config.result;

// The output laid out by matmul itself, in Dest from dest_row.
template <matmul_modifiers Mods = matmul_modifiers {}, const TensorSpec<6>& A, class RA, const TensorSpec<6>& B, class RB>
inline __attribute__((always_inline)) auto matmul(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, std::uint32_t dest_row)
{
    static_assert(accepted<matmul_op, A, B, Mods>, "matmul takes 32x32 tiles in L1 with matching K, in formats the unpacker can write to Src, no transpose yet");
    return matmul<Mods>(in0, in1, Tensor<matmul_result<A, B, Mods>>(dest_row));
}

} // namespace llk2
