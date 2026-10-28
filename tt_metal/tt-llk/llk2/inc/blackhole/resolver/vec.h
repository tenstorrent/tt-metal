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

/// The format the SFPU reads from Dest and writes back: FP32 in the 32-bit Dest mode, else BF16.
constexpr DataFormat vec_dest_format(bool dest_fp32)
{
    return dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
}

constexpr bool vec_accepts(const TensorSpec<6>& acc, const TensorSpec<6>&, const vec_call& call, const KernelSettings& s)
{
    const bool storage = acc.storage == Storage::Dest && acc.format == vec_dest_format(s.dest_fp32);
    const bool tiles   = tile_rows(acc.shape) == vec_tile_dim && tile_cols(acc.shape) == vec_tile_dim;
    const bool thread  = call.mods.thread == Thread::Math || call.mods.thread == Thread::Pack;
    return storage && tiles && call.function != VecFunction::None && thread;
}

constexpr OpConfig vec_options(const TensorSpec<6>& acc, const TensorSpec<6>&, const vec_call&, const KernelSettings& s)
{
    return {
        Exu::Sfpu,
        {Where::Dest, acc.format},
        {Where::None, DataFormat::Invalid},
        s.math_fidelity, // unused: the SFPU has no fidelity phases
        acc,             // in place
    };
}

constexpr Work vec_work(const TensorSpec<6>& acc, const TensorSpec<6>&, const OpConfig&)
{
    constexpr std::uint32_t one_step = 1; // nothing is folded into an output tile
    return {acc.shape.sizes[TileY], acc.shape.sizes[TileX], one_step, acc.shape.sizes[DatumY], faces_per_tile(acc.shape)};
}

constexpr Plan vec_plan(const TensorSpec<6>&, const TensorSpec<6>&, const vec_call& call, const OpConfig& c, const Work& k)
{
    const UnpackPlan not_unpacked {Where::None, DataFormat::Invalid, DataFormat::Invalid, 0, 0, 0, Layout::Tilized, Broadcast::None, false, 0, 0, 0};
    const UnpackWalk no_walk {0, 0, 0, Where::None};
    const MathPlan math {
        Exu::Sfpu,
        c.math_fidelity,
        c.result.format == DataFormat::Float32,
        c.in0.format,
        false, // no transpose
        static_cast<std::uint32_t>(call.function),
        call.mods.fast,
    };
    return {Op::Vec, not_unpacked, not_unpacked, no_walk, math, PackPlan {}, k};
}

inline constexpr OpApi<vec_call> vec_op {&vec_accepts, &vec_options, &vec_work, &vec_plan};

} // namespace llk2
