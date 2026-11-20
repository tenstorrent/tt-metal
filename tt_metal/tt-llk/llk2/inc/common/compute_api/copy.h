// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: copy. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's per-thread calls
// execute only on their own thread. src's L1 address is read only on the unpack thread; it needs to be valid only
// there (tensor/tensor.h).
//
//   y = copy(src, acc);                                        // y = src; src: [M, N] in L1, acc: [M, N] in Dest
//   y = copy<copy_modifiers{.transpose = true}>(src, acc);     // y = each tile of src transposed (width <-> height)
//
// Replaces today's copy_tile / copy_block (+ copy_init) and transpose_wh_tile (+ transpose_wh_init). acc is a Dest
// tensor (compute_api/dest.h); copy overwrites it in place and returns it as y: the same rows and release, the FPU's
// (math's), whichever way the tiles went, since math waits for an unpack to Dest inside copy. Tile i of src goes to
// tile i of acc. It takes src over for the unpackers and acc for the FPU first (compute_api/sync.h).
//
// The call resolves itself as a chain of one (resolver/resolver.h): accepts is checked at compile time, then the
// Plan drives llk-lib's unpack (TRISC0) and copy (TRISC1). The Resolver decides whether the tiles go through SrcA
// (math moves them to Dest) or straight to Dest (32-bit data with a 32-bit Dest); both threads follow the Plan.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/copy.h"
#include "llk_lib/sync.h"
#include "llk_lib/unpack.h"
#include "resolver/copy.h"

namespace llk
{

/// src, tile i to tile i, each tile transposed when Mods.transpose is set, written to acc and returned as math's.
template <copy_modifiers Mods = copy_modifiers {}, class S, class RS, class D, class RD>
inline __attribute__((always_inline)) auto copy(const Tensor<S, RS>& src, const Tensor<D, RD>& acc)
{
    static_assert(D::storage() == Storage::Dest, "copy writes Dest");
    static_assert(accepted<copy_op, layout_of<S>, layout_of<S>, Mods>, "copy takes 32x32 tiles in L1, in a float format the unpacker can write to Src or Dest");

    constexpr const Resolved& r = resolved<copy_op, layout_of<S>, layout_of<S>, Mods>;
    static_assert(
        stdx::ranges::equal(r.config.result.sizes(), layout_of<D>.sizes()) && r.config.result.format() == D::format(),
        "acc must have src's shape and the format copy writes to Dest (FP32 with a 32-bit Dest, else BF16 or FP16)");

    llk_lib::acquire<OwnerThread::Compute0, OwnerExu::Unpack>(src);
    llk_lib::acquire<OwnerThread::Compute1, OwnerExu::Fpu>(acc);
    llk_lib::unpack<OwnerThread::Compute0, r.plan>(src.address());
    llk_lib::copy<OwnerThread::Compute1, r.plan>(acc.address());
    return acc.with_owner(OwnerThread::Compute1, OwnerExu::Fpu);
}

} // namespace llk
