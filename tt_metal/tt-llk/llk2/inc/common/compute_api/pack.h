// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: pack. Single thread, arch-independent. Every TRISC runs this same call; llk-lib's per-thread call
// executes only on the pack thread. dst's L1 address is read only there; it needs to be valid only on the pack
// thread (tensor/tensor.h).
//
//   out = pack(src, dst);                                     // out = src; src: a Dest tensor, dst: an L1 tensor
//   out = pack<pack_modifiers{.relu = true}>(src, dst);       // out = max(src, 0)
//   out = pack<pack_modifiers{.l1_acc = Toggle::On}>(src, dst);              // out = dst + src (in dst's format, in L1)
//   out = pack<pack_modifiers{.l1_acc = Toggle::Runtime}>(src, dst, accumulate);
//                                                        // dst += src if accumulate, else dst = src; one build
//                                                        // for both (the bit is written from the argument)
//   With Toggle::Off or Toggle::On the argument may be left out; if given, it is ignored.
//
// Packs every tile of src: tile i of src (row-major) to tile i of dst. src and dst have the same shape (tile grid
// and 32x32 tiles); dst is dense in L1 and its address is the byte address of its first tile. The packer converts
// from the Dest format to dst's format; which conversions it can do, and which formats L1 can accumulate in
// (Float32, Float16_b, Float16), are checked at compile time (resolver/pack.h).
//
// out is dst, written: the same address and release (release(out) pushes a buffer's pages), now the packers'. pack
// takes src over for the packer first (compute_api/sync.h): when math wrote src, math hands it over here, so no other
// call is needed between the op that wrote src and pack. It may be called more than once on the same src (for
// example into several L1 tensors); each call takes it over again.
//
// Each call programs the packer completely (configure) and then packs (execute); there is no init, uninit or
// reconfig.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/pack.h"
#include "llk_lib/sync.h"
#include "resolver/pack.h"

namespace llk2
{

namespace detail
{
/// The checks and the per-thread call both pack forms share.
template <pack_modifiers Mods, const TensorSpec<tensor_rank>& D, const TensorSpec<tensor_rank>& O>
inline __attribute__((always_inline)) void pack_checked(std::uint32_t src_address, std::uint32_t dst_address, bool l1_acc)
{
    static_assert(D.storage == Storage::Dest, "pack reads a Dest tensor");
    static_assert(O.storage == Storage::L1, "pack writes an L1 tensor");
    static_assert(!Mods.untilize, "pack: untilize is not ported yet");
    static_assert(
        accepted<pack_op, D, O, Mods>,
        "pack takes a Dest tensor and a dense L1 tensor of the same shape in 32x32 tiles, a Dest format that matches the kernel's Dest "
        "mode, an L1 format the packer can convert to, and l1_acc (On or Runtime) only into Float32, Float16_b or Float16");

    constexpr const Resolved& r = resolved<pack_op, D, O, Mods>;
    llk_lib::acquire<Thread::Pack, Exu::Packer, D>();
    llk_lib::acquire<Thread::Pack, Exu::Packer, O>();
    llk_lib::pack<Thread::Pack, r.plan>(src_address, dst_address, l1_acc);
}
} // namespace detail

/// src (with Mods' ReLU, or added into dst with l1_acc = Toggle::On): every tile of the Dest tensor src to the L1
/// tensor dst, returned as the packers'. Mods.l1_acc must not be Toggle::Runtime here: that form takes the l1_acc
/// argument (below).
template <pack_modifiers Mods = pack_modifiers {}, const TensorSpec<tensor_rank>& D, class RD, const TensorSpec<tensor_rank>& O, class RO>
inline __attribute__((always_inline)) auto pack(const Tensor<D, RD>& src, const Tensor<O, RO>& dst)
{
    static_assert(Mods.l1_acc != Toggle::Runtime, "pack: l1_acc = Toggle::Runtime needs the run-time l1_acc argument");
    detail::pack_checked<Mods, D, O>(src.address, dst.address, false /* unused: l1_acc is Off or On */);
    return as_owned<Thread::Pack, Exu::Packer>(dst);
}

/// As above, with the run-time l1_acc argument. With l1_acc = Toggle::Runtime, src is added into dst when l1_acc is true
/// and written over dst when it is false. With Toggle::Off or Toggle::On the modifier decides and the argument is
/// ignored (no code reads it), so a kernel can pick Off or Runtime from a compile-time setting and keep one call.
template <pack_modifiers Mods, const TensorSpec<tensor_rank>& D, class RD, const TensorSpec<tensor_rank>& O, class RO>
inline __attribute__((always_inline)) auto pack(const Tensor<D, RD>& src, const Tensor<O, RO>& dst, bool l1_acc)
{
    detail::pack_checked<Mods, D, O>(src.address, dst.address, l1_acc);
    return as_owned<Thread::Pack, Exu::Packer>(dst);
}

} // namespace llk2
