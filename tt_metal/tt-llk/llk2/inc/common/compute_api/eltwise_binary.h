// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: add, sub and mul on the FPU (the form of an op: compute_api/modifiers.h). Threads: unpack, math. in0's
// and in1's L1 addresses are read on the unpack thread.
//
//   y = add(a, b, acc);                                              // y = a + b; a, b in L1, acc in Dest, same shape
//   y = add<add_modifiers{.bcast = Broadcast::Row}>(a, bias, acc);   // y = a + bias's row 0, repeated down every row
//   y = sub(a, b, acc);                                              // y = a - b
//   y = sub<sub_modifiers{.bcast = Broadcast::Col}>(s, max, acc);    // y = s - max's column 0, repeated across every column
//   y = mul(a, b, acc);                                              // y = a * b, in the kernel's fidelity
//   y = mul<mul_modifiers{.bcast = Broadcast::Col}>(o, c, acc);      // y = o * c's column 0, repeated across every column
//   y = mul<mul_modifiers{.bcast = Broadcast::Col, .accumulate = true}>(u, c, matmul(p, v, acc));
//                                                                    // y = p v + u * c's column 0: added into acc
//
// Broadcasts, for in0 of [M, N] (tiles [M/32, N/32]); every op takes all four:
//   None    in1 has in0's shape.
//   Row     in1 is one row of tiles ([32, N]: a bias); row 0 of tile column c goes down every row of in0's tile column c.
//   Col     in1 is one column of tiles ([M, 32]: per-row values in column 0, as a reduce leaves them); column 0 of
//           tile row r goes across every column of in0's tile row r.
//   Scalar  in1 is one tile ([32, 32]); its datum [0, 0] goes everywhere.
// This replaces 1.0's add_tiles / sub_tiles / mul_tiles, their *_bcast_rows / *_bcast_cols / *_bcast_scalar forms
// (BroadcastType::ROW / COL / SCALAR) and their inits.
//
// mul runs as many fidelity phases as the kernel's fidelity (LLK2_MATH_FIDELITY: LoFi 1, HiFi2 2, ..., HiFi4 4); add
// and sub always run one. All three overwrite acc: mul zeroes acc's tiles before it multiplies (ELWMUL adds to Dest).
// mul with .accumulate does not: it adds its product into what acc holds (an earlier op's result in the same Dest
// tensor), so the two results meet in Dest without a pack and an unpack in between.
//
// A size known only at run time (tensor/tensor.h, "run-time sizes"): in0 and in1 may be tensors whose spec marks the
// same tile-grid size run time (runtime_rows / runtime_cols, tensor/tensor.h) and that hold the same count. The call
// is the same; acc is a Dest tensor of the bound's size, of which the op writes the first tiles.
//
//   const auto a = wait_front<runtime_cols<a_spec>>(lhs, n);         // n tiles of a row of at most a_spec's
//   const auto b = wait_front<runtime_cols<b_spec>>(rhs, n);
//   y = add(a, b, acc);                                              // acc: a_spec's shape in Dest
//
// Result and owners. acc is a Dest tensor (compute_api/dest.h); the op overwrites it in place and returns it as y:
// the same rows and release, now the FPU's. It takes in0 and in1 over for the unpackers and acc for the FPU first
// (compute_api/sync.h).
//
// Resolver and llk-lib. add_op, sub_op, mul_op (resolver/eltwise_binary.h); llk-lib's unpack (TRISC0) and
// eltwise_binary (TRISC1). Unpack to math is Src dvalid, inside the op. The three are one family: they differ only
// in their OpApi, and share one modifier struct (add_modifiers, sub_modifiers and mul_modifiers name it) and the
// body detail::eltwise_binary.

#pragma once

#include "compute_api/counts.h"
#include "compute_api/modifiers.h"
#include "llk_lib/eltwise_binary.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/eltwise_binary.h"

namespace llk2
{

namespace detail
{
/// The body add, sub and mul share: one eltwise binary op resolved through Api, written to acc and returned as the
/// FPU's.
template <const auto& Api, eltwise_binary_modifiers Mods, const auto& A, class RA, const auto& B, class RB, const auto& D, class RD>
inline __attribute__((always_inline)) auto eltwise_binary(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& acc)
{
    static_assert(D.storage == Storage::Dest, "add, sub and mul write Dest");
    constexpr const Accepts& ok = accepted_parts<Api, A, B, Mods>;
    static_assert(ok.storage, "add, sub and mul read in0 and in1 from L1");
    static_assert(ok.tiles, "add, sub and mul take 32x32 tiles. Not ported: other tile sizes");
    static_assert(
        ok.shapes,
        "in1 has in0's shape, or with Broadcast::Row one tile row of in0's width, with Broadcast::Col one tile column of in0's height, with Broadcast::Scalar "
        "one tile");
    static_assert(ok.layout, "add, sub and mul take in0 and in1 contiguous, or as windows of a wider tile grid");
    static_assert(ok.formats, "add, sub and mul take float formats the unpacker can write to Src, with the same exponent width. Not ported: integer formats");
    static_assert(ok.modifiers, ".accumulate is mul's. Not ported: .accumulate for add and sub");
    static_assert(
        ok.runtime,
        "add, sub and mul take a run-time size as in0's tile rows, or as the tile columns of one tile row; in1 marks the same size, and none where the "
        "broadcast leaves it one tile row or column there");

    constexpr const Resolved& r = resolved<Api, A, B, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format, "acc must have in0's shape and the kernel's Dest format");

    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, A>();
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, B>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, D>();
    LLK_ASSERT(same_runtime_count(in0, in1), "in0 and in1 hold different run-time sizes");
    const auto counts = work_counts<r.work>(in0, in1); // the Work's run-time counts; empty when every size is compile time
    llk_lib::unpack<Thread::Unpack, r.plan>(counts, in0.address, in1.address);
    llk_lib::eltwise_binary<Thread::Math, r.plan>(counts, acc.address);
    return result_of<Thread::Math, Exu::Fpu, r.work>(acc, counts);
}
} // namespace detail

/// in0 + in1 (in1 broadcast as Mods.bcast says), written to acc and returned as the FPU's.
template <add_modifiers Mods = add_modifiers {}, const auto& A, class RA, const auto& B, class RB, const auto& D, class RD>
inline __attribute__((always_inline)) auto add(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& acc)
{
    return detail::eltwise_binary<add_op, Mods>(in0, in1, acc);
}

/// in0 - in1 (in1 broadcast as Mods.bcast says), written to acc and returned as the FPU's.
template <sub_modifiers Mods = sub_modifiers {}, const auto& A, class RA, const auto& B, class RB, const auto& D, class RD>
inline __attribute__((always_inline)) auto sub(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& acc)
{
    return detail::eltwise_binary<sub_op, Mods>(in0, in1, acc);
}

/// in0 * in1 (in1 broadcast as Mods.bcast says), in the kernel's fidelity, written to acc and returned as the FPU's;
/// with Mods.accumulate, added into acc.
template <mul_modifiers Mods = mul_modifiers {}, const auto& A, class RA, const auto& B, class RB, const auto& D, class RD>
inline __attribute__((always_inline)) auto mul(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1, const Tensor<D, RD>& acc)
{
    return detail::eltwise_binary<mul_op, Mods>(in0, in1, acc);
}

} // namespace llk2
