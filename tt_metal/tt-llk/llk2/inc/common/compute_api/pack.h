// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: pack (the form of an op: compute_api/modifiers.h). Thread: pack. dst's L1 address is read on the pack
// thread.
//
//   out = pack(src, dst);                                     // out = src; src: a Dest tensor, dst: an L1 tensor
//   out = pack<pack_modifiers{.relu = true}>(src, dst);       // out = max(src, 0)
//   out = pack<pack_modifiers{.l1_acc = Toggle::On}>(src, dst);              // out = dst + src (in dst's format, in L1)
//   out = pack<pack_modifiers{.l1_acc = Toggle::Runtime}>(src, dst, accumulate);
//                                                        // dst += src if accumulate, else dst = src; one build
//                                                        // for both (the bit is written from the argument)
//   With Toggle::Off or Toggle::On the argument may be left out; if given, it is ignored.
//   out = pack<pack_modifiers{.sum_tiles = true}>(src, dst);                 // out = src's tiles added up; dst: one tile
//   out = pack<pack_modifiers{.l1_acc = Toggle::On, .sum_tiles = true}>(src, dst);        // out = dst + src's tiles added up
//   out = pack<pack_modifiers{.l1_acc = Toggle::Runtime, .sum_tiles = true}>(src, dst, accumulate);
//                                                        // the same two, chosen by the argument
//
// Packs every tile of src: tile i of src (row-major) to tile i of dst. src and dst have the same shape (tile grid
// and 32x32 tiles); dst is dense in L1 and its address is the byte address of its first tile. The packer converts
// from the Dest format to dst's format; which conversions it can do, and which formats L1 can accumulate in
// (Float32, Float16_b, Float16), are checked at compile time (resolver/pack.h).
//
// sum_tiles: dst is one 32x32 tile, whatever src's tile grid, and every tile of src goes to it: the first as l1_acc
// says (Off: over dst, On: added to dst, Runtime: the argument), every later one added to dst by the packer's L1
// accumulation. dst ends as the elementwise sum of src's tiles (plus what it held, when the first tile accumulated),
// rounded in dst's format after every tile, so dst's format must be one L1 can accumulate in. ReLU applies to every
// tile before it is added. It gives the sum of a block's tiles without sending them through an unpacker and the FPU
// again: a tensor is packed to its place and then, from the same Dest tensor, into its sum tile (a softmax's
// denominator, taken from the exp block).
//
// A size known only at run time (tensor/tensor.h, "run-time sizes"): tile rows, or the tile columns of one tile row.
// src holds it when the op that wrote src ran on a run-time size (the op's result carries the count): pack packs
// that many tiles, to a dst that marks the same size and holds the same count, or with sum_tiles into dst's one tile.
// Or dst alone holds it: src is then a whole Dest tensor of the bound's size, of which pack packs the first tiles.
//   out = pack(add(a, b, acc), reserve_back<runtime_cols<out_spec>>(out_buffer, n));    // a, b: n tiles each
//   out = pack(src, reserve_back<runtime_cols<out_spec>>(out_buffer, n));                // the first n tiles of src
//
// Result and owners. out is dst, written: the same address and release (release(out) pushes a buffer's pages), now
// the packers'. pack takes src over for the packer first (compute_api/sync.h): when math wrote src, math hands it over
// here, so no other call is needed between the op that wrote src and pack. It may be called more than once on the
// same src (for example into several L1 tensors); each call takes it over again.
//
// Resolver and llk-lib. pack_op (resolver/pack.h); llk-lib's pack (TRISC2). Each call programs the packer completely
// (configure) and then packs (execute); there is no init, uninit or reconfig. Where pack differs from the other ops,
// and why:
//   - The Resolver's second operand is dst, the tensor pack writes, not a second input: the L1 format is the caller's
//     choice and decides the Plan, where an FPU op's result format follows from its inputs and the Dest mode. So
//     accepts checks dst, and there is no separate check of the result against the output.
//   - l1_acc = Toggle::Runtime adds a run-time argument, accumulate, after the tensors as every run-time argument is
//     (compute_api/modifiers.h). The two forms share the body detail::pack.

#pragma once

#include <cstdint>

#include "compute_api/counts.h"
#include "compute_api/modifiers.h"
#include "llk_lib/pack.h"
#include "llk_lib/sync.h"
#include "resolver/pack.h"

namespace llk2
{

namespace detail
{
/// The body both forms of pack share. `accumulate` is read only with Mods.l1_acc = Toggle::Runtime.
template <pack_modifiers Mods, const auto& D, class RD, const auto& O, class RO>
inline __attribute__((always_inline)) auto pack(const Tensor<D, RD>& src, const Tensor<O, RO>& dst, bool accumulate)
{
    constexpr const Accepts& ok = accepted_parts<pack_op, D, O, Mods>;
    static_assert(ok.storage, "pack reads Dest and writes L1");
    static_assert(ok.tiles, "pack takes 32x32 tiles. Not ported: other tile sizes");
    static_assert(ok.shapes, "pack takes src and dst of the same shape; with sum_tiles dst is one tile");
    static_assert(ok.layout, "pack takes a whole Dest tensor and a dense L1 tensor. Not ported: a window of a wider tensor as dst");
    static_assert(
        ok.formats,
        "pack takes a Dest format of the kernel's Dest mode (Float32 or Int32 with a 32-bit Dest, else Float16_b, Float16 or UInt16) and an L1 format the "
        "packer can convert it to");
    static_assert(ok.modifiers, "pack adds into L1 (l1_acc On or Runtime, or sum_tiles) only in Float32, Float16_b or Float16. Not ported: untilize");
    static_assert(
        ok.runtime,
        "pack takes a run-time size as tile rows, or as the tile columns of one tile row: src's, as the op that wrote it returned it, and then also dst's "
        "unless sum_tiles makes dst one tile; or dst's alone, src being a whole Dest tensor of the bound's size");

    constexpr const Resolved& r = resolved<pack_op, D, O, Mods>;
    llk_lib::acquire<Thread::Pack, Exu::Packer, D>();
    llk_lib::acquire<Thread::Pack, Exu::Packer, O>();
    LLK_ASSERT(same_runtime_count(src, dst), "pack: src was written with another run-time size than dst has");
    const auto counts = work_counts<r.work>(src, dst); // the Work's run-time counts; empty when every size is compile time
    llk_lib::pack<Thread::Pack, r.plan>(counts, src.address, dst.address, accumulate);
    return as_owned<Thread::Pack, Exu::Packer>(dst); // dst as it came: its count is the one packed (same_runtime_count)
}
} // namespace detail

/// src (with Mods' ReLU, L1 accumulation or tile sum): every tile of the Dest tensor src to the L1 tensor dst, returned
/// as the packers'. Mods.l1_acc is Toggle::Off (src is written over dst) or Toggle::On (src is added into dst).
template <pack_modifiers Mods = pack_modifiers {}, const auto& D, class RD, const auto& O, class RO>
inline __attribute__((always_inline)) auto pack(const Tensor<D, RD>& src, const Tensor<O, RO>& dst)
{
    static_assert(Mods.l1_acc != Toggle::Runtime, "pack: l1_acc = Toggle::Runtime needs the run-time argument: pack<Mods>(src, dst, accumulate)");
    return detail::pack<Mods>(src, dst, false /* accumulate: not read, the modifier is Off or On */);
}

/// pack with the run-time argument, for Mods.l1_acc = Toggle::Runtime: src is added into dst when `accumulate` is true
/// and written over dst when it is false. With Toggle::Off or Toggle::On the modifier decides and the argument is
/// ignored (no code reads it), so a kernel can pick Off or Runtime from a compile-time setting and keep one call.
template <pack_modifiers Mods, const auto& D, class RD, const auto& O, class RO>
inline __attribute__((always_inline)) auto pack(const Tensor<D, RD>& src, const Tensor<O, RO>& dst, bool accumulate)
{
    return detail::pack<Mods>(src, dst, accumulate);
}

} // namespace llk2
