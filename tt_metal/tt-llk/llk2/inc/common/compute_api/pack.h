// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: pack. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's per-thread call
// executes only on the pack thread. dst's L1 address is read only there; it needs to be valid only on the pack
// thread (tensor/tensor.h).
//
//   pack(src, dst);                                     // dst = src; src: a Dest tensor, dst: an L1 tensor
//   pack<pack_modifiers{.relu = true}>(src, dst);       // dst = max(src, 0)
//   pack<pack_modifiers{.l1_acc = true}>(src, dst);     // dst += src (in dst's format, in L1)
//
// Packs every tile of src: tile i of src (row-major) to tile i of dst. src and dst have the same shape (tile grid
// and 32x32 tiles); dst is dense in L1 and its address is the byte address of its first tile. The packer converts
// from the Dest format to dst's format; which conversions it can do, and which formats L1 can accumulate in
// (Float32, Float16_b, Float16), are checked at compile time (resolver/pack.h).
//
// Call it between dest_commit and dest_release (compute_api/dest.h):
//   llk2::dest_commit(acc);
//   llk2::pack(acc, out);
//   llk2::dest_release(acc);
// It may be called more than once per section (for example on parts of acc, or into several L1 tensors).
//
// Each call programs the packer completely (configure) and then packs (execute); there is no init, uninit or
// reconfig.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/pack.h"
#include "resolver/pack.h"

namespace llk2
{

/// dst = src (with Mods' ReLU, or added into dst with l1_acc): every tile of the Dest tensor src to the L1 tensor dst.
template <pack_modifiers Mods = pack_modifiers {}, const TensorSpec<tensor_rank>& D, const TensorSpec<tensor_rank>& O>
inline __attribute__((always_inline)) void pack(const Tensor<D>& src, const Tensor<O>& dst)
{
    static_assert(D.storage == Storage::Dest, "pack reads a Dest tensor");
    static_assert(O.storage == Storage::L1, "pack writes an L1 tensor");
    static_assert(!Mods.untilize, "pack: untilize is not ported yet");
    static_assert(
        accepted<pack_op, D, O, Mods>,
        "pack takes a Dest tensor and a dense L1 tensor of the same shape in 32x32 tiles, a Dest format that matches the kernel's Dest "
        "mode, an L1 format the packer can convert to, and l1_acc only into Float32, Float16_b or Float16");

    constexpr const Resolved& r = resolved<pack_op, D, O, Mods>;
    llk_lib::pack<Thread::Pack, r.plan>(src.address, dst.address);
}

} // namespace llk2
