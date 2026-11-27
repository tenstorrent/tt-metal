// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: copy (the form of an op: compute_api/modifiers.h). Threads: unpack, math. src's L1 address is read on
// the unpack thread.
//
//   y = copy(src, acc);                                        // y = src; src: [M, N] in L1, acc: [M, N] in Dest
//   y = copy<copy_modifiers{.transpose = true}>(src, acc);     // y = each tile of src transposed (width <-> height)
//
// Tile i of src goes to tile i of acc. Replaces 1.0's copy_tile / copy_block (+ copy_init) and transpose_wh_tile
// (+ transpose_wh_init).
//
// Result and owners. acc is a Dest tensor (compute_api/dest.h); copy overwrites it in place and returns it as y: the
// same rows and release, the FPU's (math's), whichever way the tiles went, since math waits for an unpack to Dest
// inside copy. It takes src over for the unpackers and acc for the FPU first (compute_api/sync.h).
//
// Resolver and llk-lib. copy_op (resolver/copy.h); llk-lib's unpack (TRISC0) and copy (TRISC1). The Resolver decides
// whether the tiles go through SrcA (math moves them to Dest) or straight to Dest (32-bit data with a 32-bit Dest);
// both threads follow the Plan. copy has one input: the Resolver's second operand is src again, and is not read.
// copy takes no run-time size: its counts are always empty.

#pragma once

#include "compute_api/counts.h"
#include "compute_api/modifiers.h"
#include "llk_lib/copy.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/copy.h"

namespace llk2
{

/// src, tile i to tile i, each tile transposed when Mods.transpose is set, written to acc and returned as math's.
template <copy_modifiers Mods = copy_modifiers {}, const auto& A, class RA, const auto& D, class RD>
inline __attribute__((always_inline)) auto copy(const Tensor<A, RA>& src, const Tensor<D, RD>& acc)
{
    static_assert(D.storage == Storage::Dest, "copy writes Dest");
    constexpr const Accepts& ok = accepted_parts<copy_op, A, A, Mods>;
    static_assert(ok.storage, "copy reads src from L1");
    static_assert(ok.tiles, "copy takes 32x32 tiles. Not ported: other tile sizes");
    static_assert(ok.layout, "copy takes a contiguous tensor. Not ported: a window of a wider tensor");
    static_assert(ok.formats, "copy takes a float format the unpacker can write to Src or Dest. Not ported: integer formats");
    static_assert(ok.runtime, "copy takes no run-time size (not ported)");

    constexpr const Resolved& r = resolved<copy_op, A, A, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
        "acc must have src's shape and the format copy writes to Dest (FP32 with a 32-bit Dest, else BF16 or FP16)");

    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, A>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, D>();
    const auto counts = work_counts<r.work>(src, src); // the Work's run-time counts; empty: copy takes no run-time size
    llk_lib::unpack<Thread::Unpack, r.plan>(counts, src.address);
    llk_lib::copy<Thread::Math, r.plan>(counts, acc.address);
    return result_of<Thread::Math, Exu::Fpu, r.work>(acc, counts);
}

} // namespace llk2
