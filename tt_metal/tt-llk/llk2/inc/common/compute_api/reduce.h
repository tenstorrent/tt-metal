// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: reduce on the FPU (the form of an op: compute_api/modifiers.h). Threads: unpack, math. in's and
// scaler's L1 addresses are read on the unpack thread.
//
//   y = reduce(s, scaler, acc);                                          // y = row max of s, into column 0
//   y = reduce<reduce_modifiers{.kind = ReduceKind::Sum}>(s, scaler, acc); // y = row sum of s (times the scaler)
//   y = reduce(s, scaler, m_prev, acc);                                  // y = max(m_prev, row max of s): seeded
//
// in is [32 R, 32 N]: R tile rows of N tiles. Each row of in is folded over all N tiles of its tile row, into column 0
// of that tile row's output tile; acc and y are [32 R, 32], every datum but column 0 is 0 (what 1.0's
// reduce_tile / reduce_block_max_row leave after the pack reduce mask). scaler is one tile whose row 0 of face 0 is
// 1.0 for a plain max or sum (1/N for a mean). This replaces 1.0's reduce_init /
// reduce_tile / reduce_uninit for REDUCE_ROW and the block max-row reduce SDPA uses (reduce_block_max_row*, with no
// pack reduce mask config / clear: the output in Dest is already clean).
//
// The Dest mode. acc is in the kernel's Dest format. With a 32-bit Dest a max is FP32 with 19 bits kept (TF32, what a
// Src register holds; the low 13 bits are 0), which is every bit of a max. A seed reaches acc through SrcA, so a
// Float32 seed is cut to 19 bits too. A sum there is computed as a product with the scaler (resolver/reduce.h), which
// keeps Dest's width; the scaler's row 0 must then be 1.0 over all 32 columns (a reduce scaler tile is).
//
// Seeded (a running max, m_new = max(m_prev, rowmax(S))): seed is a previous result, [32 R, 32] in L1, its
// values in column 0. The four-tensor form copies it into acc transposed (copy<copy_modifiers {.transpose = true}>),
// which puts it in the rows the reduction folds into, then reduces with .accumulate = true. 1.0's SDPA does the same
// with copy_tile_custom (a within-face transpose) before reduce_block_max_row. A kernel that loads the seed itself
// passes .accumulate = true with the three-tensor form; acc must then hold that transposed copy of a result-shaped
// tensor. A sum in a 32-bit Dest adds into column 0 itself: there the seed is copied as it is, and it must be what a
// reduce leaves, 0 outside column 0 (the pooled forms read the seed's column 0 only).
// The four-tensor form is the one Compute API call made of two ops (copy, then reduce): the seed has to be in acc in
// a layout only reduce knows (transposed), so the pair is kept behind one call instead of in every kernel.
//
// Result and owners. acc is a Dest tensor (compute_api/dest.h); the op overwrites it in place and returns it as y:
// the same rows and release, now the FPU's. It takes in and scaler over for the unpackers and acc for the FPU first
// (compute_api/sync.h).
//
// Resolver and llk-lib. reduce_op (resolver/reduce.h); llk-lib's unpack (TRISC0) and reduce (TRISC1). Unpack to math
// is Src dvalid, inside the op. Only in may hold a run-time size (its tile columns: the row length), and the result
// is one tile per tile row whatever that length, so the result carries no count.

#pragma once

#include "compute_api/copy.h"
#include "compute_api/counts.h"
#include "compute_api/modifiers.h"
#include "llk_lib/reduce.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/reduce.h"

namespace llk2
{

/// The Mods.kind reduction of every row of in (over all its tile row's tiles) into column 0 of acc, returned as the
/// FPU's. With Mods.accumulate, folded into the seed acc already holds (see the top of this file).
template <reduce_modifiers Mods = reduce_modifiers {}, const auto& A, class RA, const auto& B, class RB, const auto& D, class RD>
inline __attribute__((always_inline)) auto reduce(const Tensor<A, RA>& in, const Tensor<B, RB>& scaler, const Tensor<D, RD>& acc)
{
    static_assert(D.storage == Storage::Dest, "reduce writes Dest");
    constexpr const Accepts& ok = accepted_parts<reduce_op, A, B, Mods>;
    static_assert(ok.storage, "reduce reads in and the scaler from L1");
    static_assert(ok.tiles, "reduce takes 32x32 tiles. Not ported: other tile sizes");
    static_assert(ok.shapes, "reduce's scaler is one tile");
    static_assert(ok.layout, "reduce takes in contiguous, or as a window of a wider tile grid, and a contiguous scaler");
    static_assert(ok.formats, "reduce takes in and the scaler in an 8-bit-exponent float format (BF16, BFP8_b, ...). Not ported: 5-bit-exponent formats");
    static_assert(ok.modifiers, "reduce reduces rows, to their max or their sum. Not ported: column and scalar reduce");
    static_assert(ok.runtime, "reduce takes a run-time size as in's tile columns (the row length), and none on the scaler");

    constexpr const Resolved& r = resolved<reduce_op, A, B, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
        "acc must be one tile per tile row of in ([32 R, 32]) in the kernel's Dest format");

    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, A>();
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, B>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, D>();
    const auto counts = work_counts<r.work>(in, scaler); // the Work's run-time counts; empty when every size is compile time
    llk_lib::unpack<Thread::Unpack, r.plan>(counts, in.address, scaler.address);
    llk_lib::reduce<Thread::Math, r.plan>(counts, acc.address);
    return as_owned<Thread::Math, Exu::Fpu>(acc); // one tile per tile row, whatever the run-time row length: no count to carry
}

/// Seeded: the Mods.kind reduction of every row of in folded into seed's column 0 (a previous result), written to acc
/// and returned as the FPU's. seed is copied into acc transposed first, then reduced with .accumulate = true.
template <reduce_modifiers Mods = reduce_modifiers {}, const auto& A, class RA, const auto& B, class RB, const auto& C, class RC, const auto& D, class RD>
inline __attribute__((always_inline)) auto reduce(const Tensor<A, RA>& in, const Tensor<B, RB>& scaler, const Tensor<C, RC>& seed, const Tensor<D, RD>& acc)
{
    // The shape only: copy checks that seed is in L1.
    static_assert(equal(C.shape.sizes, D.shape.sizes), "seed must have acc's shape (it is a previous result)");
    // This form folds into the seed whatever Mods.accumulate says.
    constexpr reduce_modifiers seeded {.kind = Mods.kind, .dim = Mods.dim, .accumulate = true};
    // Where the reduction folds into: the pooled rows (the seed transposed), or for a sum by product column 0 itself.
    constexpr copy_modifiers to_folded_rows {.transpose = !resolved<reduce_op, A, B, seeded>.plan.math.by_product};
    return reduce<seeded>(in, scaler, copy<to_folded_rows>(seed, acc));
}

} // namespace llk2
