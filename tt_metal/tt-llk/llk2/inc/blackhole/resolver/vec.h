// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The OpApi of the SFPU functions (vec_silu, vec_tanh, vec_exp, ...) on Blackhole. One operand: a Dest tensor the SFPU
// rewrites in place, tile by tile. Nothing is unpacked, so the Plan's unpack streams are None and its walk is
// empty; MathPlan.vec names the function, MathPlan.vec_form its algorithm, MathPlan.vec_region the datums of each
// tile it rewrites and MathPlan.scale the exp's compile-time input scale.
//
//   accepts   acc in Dest, in BF16 or, with a 32-bit Dest, FP32 (dest_format_for), in 32x32 tiles; a function, in a
//             form and on a region it has (vec_has_form, vec_has_region), with the input clamping that form has and
//             a scale only for exp; driven from math or pack
//   options   the SFPU reads and writes Dest in the Dest format; the result is acc itself
//   work      the tile grid and the tile geometry
//   plan      the function, its form and region, the scale, the Dest mode
//
// A vec_ call has one tensor; the Resolver's second operand is the same tensor and is not read. The calls share this
// one OpApi, so what they hand it is a vec_call (resolver/modifiers.h) in place of a modifier struct.
//
// Not ported: tiles other than 32x32, Dest formats other than the kernel's (dest_format_for: FP16 and the integers);
// the fast exp with input clamping and on a region; the Polynomial form of exp on All or LeftFaces, and of any other
// function; LeftFacesEvenColumns for a function other than exp.

#pragma once

#include <cstdint>

#include "resolver/modifiers.h" // vec_call
#include "resolver/resolver.h"

namespace llk2
{

/// Whether function f has the form: llk_lib/vec/<function>.h has a body for it. Every function has Exact; Fast is a
/// second body some have; Polynomial is exp's (llk_lib/vec/exp.h).
constexpr bool vec_has_form(VecFunction f, VecForm form)
{
    if (f == VecFunction::None)
    {
        return false;
    }
    if (form == VecForm::Fast)
    {
        return f == VecFunction::Tanh || f == VecFunction::Gelu || f == VecFunction::GeluTanh || f == VecFunction::Sigmoid || f == VecFunction::Exp ||
               f == VecFunction::Reciprocal;
    }
    if (form == VecForm::Polynomial)
    {
        return f == VecFunction::Exp;
    }
    return true;
}

/// Whether the form of function f runs on the region. All and LeftFaces are llk-lib's walk over a tile's faces
/// (llk_lib/vec.h), the same for every function whose body is one face's iterations: every form but the fast exp,
/// which walks the whole tile in one call, and exp's Polynomial, whose body is a half face. LeftFacesEvenColumns is
/// that half-face body: exp's Exact and Polynomial forms have one.
constexpr bool vec_has_region(VecFunction f, VecForm form, VecRegion region)
{
    const bool whole_tile_body = f == VecFunction::Exp && form == VecForm::Fast;
    if (region == VecRegion::LeftFacesEvenColumns)
    {
        return f == VecFunction::Exp && (form == VecForm::Exact || form == VecForm::Polynomial);
    }
    if (form == VecForm::Polynomial)
    {
        return false;
    }
    return region == VecRegion::All || !whole_tile_body;
}

/// Whether the form of function f takes the input clamping asked for: the fast exp has no clamping, every other form
/// is right on its whole input.
constexpr bool vec_has_clamping(VecFunction f, VecForm form, bool clamp_input)
{
    return clamp_input != (f == VecFunction::Exp && form == VecForm::Fast);
}

/// accepts: see the file comment.
constexpr Accepts vec_accepts(const Spec& acc, const Spec&, const vec_call& call, const KernelSettings& settings)
{
    const vec_modifiers& mods = call.mods;
    const bool storage        = acc.storage == Storage::Dest;
    const bool tiles          = has_default_tiles(acc.shape);
    const bool formats        = acc.format == dest_format_for(settings); // the other Dest formats of the mode: not ported
    const bool form           = vec_has_form(call.function, mods.form) && vec_has_region(call.function, mods.form, mods.region) &&
                      vec_has_clamping(call.function, mods.form, mods.clamp_input);
    const bool scale     = mods.input_scale == vec_fp32_one || call.function == VecFunction::Exp;
    const bool modifiers = form && scale && (mods.thread == Thread::Math || mods.thread == Thread::Pack);
    // A run-time size (resolver/work_counts.h): tile rows, or the tile columns of one tile row (the tiles follow each other).
    const bool runtime = !is_runtime_size(acc, TileX) || acc.shape.sizes[TileY] == 1;
    return {.storage = storage, .tiles = tiles, .formats = formats, .modifiers = modifiers, .runtime = runtime};
}

/// options: the SFPU reads and writes Dest in the Dest format; the result is acc itself.
constexpr OpConfig vec_options(const Spec& acc, const Spec&, const vec_call&, const KernelSettings&)
{
    return {
        .in0           = {.where = Where::Dest, .format = acc.format},
        .in1           = {.where = Where::None, .format = DataFormat::Invalid},
        .math_fidelity = math_fidelity_lofi, // no fidelity phases
        .result        = acc,                // in place
    };
}

/// work: the tile grid and the tile geometry; no K.
constexpr Work vec_work(const Spec& acc, const Spec&, const OpConfig&)
{
    return in0_tile_grid_work(acc);
}

/// plan: nothing unpacked, the function, its form and its region on the SFPU.
constexpr Plan vec_plan(const Spec&, const Spec&, const vec_call& call, const OpConfig& config, const Work& work)
{
    const MathPlan math {
        .math_fidelity   = config.math_fidelity,
        .dest_fp32       = is_32bit_format(config.result.format),
        .format          = config.in0.format,
        .transpose_faces = false,
        .vec             = call.function,
        .vec_form        = call.mods.form,
        .vec_region      = call.mods.region,
        .scale           = call.mods.input_scale,
    };
    return {.in0 = {}, .in1 = {}, .walk = {}, .math = math, .pack = {}, .work = work};
}

inline constexpr OpApi<vec_call> vec_op {&vec_accepts, &vec_options, &vec_work, &vec_plan};

} // namespace llk2
