// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: matmul (the form of an op: compute_api/modifiers.h). Threads: unpack, math. in0's and in1's L1
// addresses are read on the unpack thread.
//
//   y = matmul(in0, in1, acc);                          // y = acc + in0 * in1; in0: [M, K], in1: [K, N], acc: [M, N] in Dest
//   y = matmul<matmul_modifiers{...}>(in0, in1, acc);
//   y = matmul<matmul_modifiers{.transpose_in1 = true}>(in0, in1, acc);  // y = acc + in0 * in1', in1' = in1 with each
//                                                                         // tile transposed (tile grid unchanged): a product with a transpose, as Q K^T
//
// Result and owners. acc is a Dest tensor (compute_api/dest.h). matmul adds into it in place and returns it as y: the
// same rows and release, now the FPU's. Before it starts, matmul takes its tensors over from their owners
// (compute_api/sync.h): in0 and in1 for the unpackers, acc for the FPU.
// matmul adds into what acc holds (.accumulate, its default and its only form: the FPU's MVMUL accumulates). A Dest
// ring's section starts at zero, so the first matmul into it is the plain product. mul and reduce say the same with
// the same modifier, .accumulate.
//
// Resolver and llk-lib. matmul_op (resolver/matmul.h); llk-lib's unpack (TRISC0) and matmul (TRISC1). Unpack to math
// is Src dvalid, inside the op.

#pragma once

#include "compute_api/counts.h"
#include "compute_api/modifiers.h"
#include "llk_lib/matmul.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/matmul.h"

namespace llk2
{

/// acc + in0 * in1, every K step, written to acc and returned as the FPU's. in0 [M, K] and in1 [K, N] in L1, acc [M, N]
/// in Dest.
template <matmul_modifiers Mods = matmul_modifiers {}, const auto& A, class RA, const auto& B, class RB, const auto& D, class RD>
inline __attribute__((always_inline)) auto matmul(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& acc)
{
    static_assert(D.storage == Storage::Dest, "matmul writes Dest");
    constexpr const Accepts& ok = accepted_parts<matmul_op, A, B, Mods>;
    static_assert(ok.storage, "matmul reads in0 and in1 from L1");
    static_assert(ok.tiles, "matmul takes 32x32 tiles. Not ported: other tile sizes");
    static_assert(ok.shapes, "matmul takes in0 [M, K] and in1 [K, N]: in0's tile columns are in1's tile rows");
    static_assert(ok.layout, "matmul takes in0 and in1 contiguous, or as windows of a wider tile grid");
    static_assert(ok.formats, "matmul takes formats the unpacker can write to Src, both float or both integer");
    static_assert(ok.modifiers, "matmul adds into acc. Not ported: .accumulate = false, a matmul that overwrites acc");
    static_assert(ok.dest_mode, "matmul of integers needs a 32-bit Dest (LLK2_DEST_FP32)");
    static_assert(
        ok.runtime,
        "matmul takes a run-time size as K (in0's tile columns and in1's tile rows, both marked), or as N (in1's tile columns) when in0 is one tile row");

    constexpr const Resolved& r = resolved<matmul_op, A, B, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
        "acc must have matmul's output shape [M, N] and the kernel's Dest format");

    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, A>();
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, B>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, D>();
    LLK_ASSERT(same_runtime_count(in0, in1), "in0 and in1 hold different run-time K");
    const auto counts = work_counts<r.work>(in0, in1); // the Work's run-time counts; empty when every size is compile time
    llk_lib::unpack<Thread::Unpack, r.plan>(counts, in0.address, in1.address);
    llk_lib::matmul<Thread::Math, r.plan>(counts, acc.address);
    return result_of<Thread::Math, Exu::Fpu, r.work>(acc, counts);
}

} // namespace llk2
