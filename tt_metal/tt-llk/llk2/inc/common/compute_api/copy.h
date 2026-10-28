// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: copy. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's per-thread calls
// execute only on their own thread. src's L1 address is read only on the unpack thread; it needs to be valid only
// there (tensor/tensor.h).
//
//   copy(src, acc);                                        // acc = src; src: [M, N] in L1, acc: [M, N] in Dest
//   copy<copy_modifiers{.transpose = true}>(src, acc);     // acc = each tile of src transposed (width <-> height)
//
// Replaces today's copy_tile / copy_block (+ copy_init) and transpose_wh_tile (+ transpose_wh_init). acc is a Dest
// tensor the caller holds (from the Dest sync API); copy overwrites it in place and returns nothing. Tile i of src
// goes to tile i of acc.
//
// The call resolves itself as a chain of one (resolver/resolver.h): accepts is checked at compile time, then the
// Plan drives llk-lib's unpack (TRISC0) and copy (TRISC1). The Resolver decides whether the tiles go through SrcA
// (math moves them to Dest) or straight to Dest (32-bit data with a 32-bit Dest); both threads follow the Plan.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/copy.h"
#include "llk_lib/unpack.h"
#include "resolver/copy.h"

namespace llk2
{

/// acc = src, tile i to tile i, each tile transposed when Mods.transpose is set.
template <copy_modifiers Mods = copy_modifiers {}, const TensorSpec<tensor_rank>& S, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void copy(const Tensor<S>& src, const Tensor<D>& acc)
{
    static_assert(D.storage == Storage::Dest, "copy writes Dest");
    static_assert(accepted<copy_op, S, S, Mods>, "copy takes 32x32 tiles in L1, in a float format the unpacker can write to Src or Dest");

    constexpr const Resolved& r = resolved<copy_op, S, S, Mods>;
    static_assert(
        equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
        "acc must have src's shape and the format copy writes to Dest (FP32 with a 32-bit Dest, else BF16 or FP16)");

    constexpr std::uint32_t no_second_input = 0; // copy has one input; unpacker 1 idles
    llk_lib::unpack<Thread::Unpack, r.plan>(src.address, no_second_input);
    llk_lib::copy<Thread::Math, r.plan>(acc.address);
}

} // namespace llk2
