// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The OpApi of the SFPU activations (vec_silu, vec_tanh, ...) on Blackhole. One operand: a Dest tensor the SFPU
// rewrites in place, tile by tile. Nothing is unpacked, so the Plan's unpack streams are None and its walk is
// empty; MathPlan.function names the activation and MathPlan.fast picks its approximate form.
//
//   vec_accepts   the operand is in Dest, in the kernel's Dest format, in 32x32 tiles, driven from math or pack
//   vec_options   the SFPU reads and writes Dest in the Dest format; the result is the operand itself
//   vec_work      tiles along y and x, faces per tile
//   vec_plan      Exu::Sfpu, the function, approximate mode, the Dest mode
//
// in1 is not used: the Compute API passes the operand twice.

#pragma once

#include "resolver/resolver.h"
#include "resolver/vec_function.h"

namespace llk2
{

inline constexpr std::uint32_t vec_tile_dim = default_tile_dim; ///< the SFPU activations are ported for 32x32 tiles only

/// accepts: see the file comment.
constexpr bool vec_accepts(const TensorSpec<tensor_rank>& acc, const TensorSpec<tensor_rank>&, const vec_call& call, const KernelSettings& s)
{
    const bool storage = acc.storage == Storage::Dest && acc.format == dest_format_for(s);
    const bool tiles   = tile_rows(acc.shape) == vec_tile_dim && tile_cols(acc.shape) == vec_tile_dim;
    const bool thread  = call.mods.thread == Thread::Math || call.mods.thread == Thread::Pack;
    return storage && tiles && call.function != VecFunction::None && thread;
}

/// options: the SFPU reads and writes Dest in the Dest format; the result is acc itself.
constexpr OpConfig vec_options(const TensorSpec<tensor_rank>& acc, const TensorSpec<tensor_rank>&, const vec_call&, const KernelSettings& s)
{
    return {
        Exu::Sfpu,
        {Where::Dest, acc.format},
        {Where::None, DataFormat::Invalid},
        s.math_fidelity, // unused: the SFPU has no fidelity phases
        acc,             // in place
    };
}

/// work: the tile grid and the tile geometry; no K.
constexpr Work vec_work(const TensorSpec<tensor_rank>& acc, const TensorSpec<tensor_rank>&, const OpConfig&)
{
    return {acc.shape.sizes[TileY], acc.shape.sizes[TileX], no_k_steps, acc.shape.sizes[DatumY], faces_per_tile(acc.shape)};
}

/// plan: nothing unpacked, the function and its mode on the SFPU.
constexpr Plan vec_plan(const TensorSpec<tensor_rank>&, const TensorSpec<tensor_rank>&, const vec_call& call, const OpConfig& c, const Work& k)
{
    const MathPlan math {
        Exu::Sfpu,
        c.math_fidelity,
        is_32bit_format(c.result.format),
        c.in0.format,
        false, // no transpose
        static_cast<std::uint32_t>(call.function),
        call.mods.fast,
    };
    return {Op::Vec, UnpackPlan {}, UnpackPlan {}, UnpackWalk {}, math, PackPlan {}, k};
}

inline constexpr OpApi<vec_call> vec_op {&vec_accepts, &vec_options, &vec_work, &vec_plan};

} // namespace llk2
