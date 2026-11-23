// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Copy's OpApi on Blackhole: an L1 tensor into a Dest tensor, tile by tile, optionally transposed (width <-> height).
// Copy has one input; the Resolver's second operand is the same tensor and is not read.
//
//   accepts   src in L1 in 32x32 tiles, contiguous, in a float format the unpacker can write to Src, or straight to
//             Dest
//   options   unpacker 0 to SrcA (math moves the tiles) or straight to Dest; the result is src's shape in Dest
//   work      one output tile per src tile
//   plan      one run over every tile of src in L1 order; math moves, or only waits
//
// Two ways to reach Dest, decided here and nowhere else:
//   through SrcA   unpacker 0 writes the tile to SrcA, math moves it to Dest (MOVA2D). Every format the unpacker
//                  can write to Src, and the only way with transpose
//   straight       unpacker 0 writes the tile to Dest itself; math only waits for it. Only for 32-bit L1 formats
//                  with a 32-bit Dest, where the trip through 19-bit SrcA would lose bits (Float32 -> TF32).
//                  This is the test harness's unpack_to_dest rule: input is 32-bit and dest_acc is on
// With transpose, the unpacker transposes inside each face (Haloize_mode) and math writes the faces back in
// transposed order (face 1 <-> face 2). The unpacker cannot transpose on the way to Dest (ISA: UNPACR to Dest with
// Haloize_mode is undefined), so a transposed 32-bit copy goes through SrcA as TF32.
//
// Not ported: integer formats (through SrcA they need the integer datapath; to Dest only INT32 / UINT32 work), the
// lossless 32-bit transpose in Dest (llk_math_transpose_dest.h), broadcasts, tiles other than 32x32, a window of a
// wider tile grid as src.

#pragma once

#include <cstdint>

#include "resolver/formats.h" // src_register_format, is_integer_format, has_5bit_exponent
#include "resolver/modifiers.h"
#include "resolver/resolver.h"

namespace llk2
{

/// True when copy unpacks straight to Dest instead of through SrcA.
constexpr bool copy_unpacks_to_dest(DataFormat l1, const copy_modifiers& mods, const KernelSettings& settings)
{
    return is_32bit_format(l1) && settings.dest_fp32 && !mods.transpose;
}

/// The Dest format a copy of l1 produces.
constexpr DataFormat copy_dest_format(DataFormat l1, const copy_modifiers& mods, const KernelSettings& settings)
{
    if (copy_unpacks_to_dest(l1, mods, settings))
    {
        return l1; // the unpacker writes FP32 / INT32 / UINT32 unchanged
    }
    if (settings.dest_fp32)
    {
        return DataFormat::Float32; // MOVA2D widens BF16 / TF32 to FP32 when Dest is 32-bit
    }
    return has_5bit_exponent(l1) ? DataFormat::Float16 : DataFormat::Float16_b;
}

/// accepts: see the file comment.
constexpr Accepts copy_accepts(const Spec& src, const Spec&, const copy_modifiers& mods, const KernelSettings& settings)
{
    const bool storage = src.storage == Storage::L1;
    const bool tiles   = has_default_tiles(src.shape);
    const bool layout  = is_contiguous(src.shape); // steps come from sizes, not strides
    // Integers: not ported.
    const bool formats = !is_integer_format(src.format) &&
                         (copy_unpacks_to_dest(src.format, mods, settings) || src_register_format(src.format, settings.dest_fp32) != DataFormat::Invalid);
    const bool runtime = !has_runtime_size(src); // a run-time size: not ported (resolver/work_counts.h)
    return {.storage = storage, .tiles = tiles, .layout = layout, .formats = formats, .runtime = runtime};
}

/// options: unpacker 0 to SrcA (math moves the tiles) or straight to Dest; the result is src's shape in Dest.
constexpr OpConfig copy_options(const Spec& src, const Spec&, const copy_modifiers& mods, const KernelSettings& settings)
{
    const bool to_dest = copy_unpacks_to_dest(src.format, mods, settings);
    return {
        .in0           = to_dest ? OperandRep {.where = Where::Dest, .format = src.format}
                                 : OperandRep {.where = Where::SrcA, .format = src_register_format(src.format, settings.dest_fp32)},
        .in1           = {.where = Where::None, .format = DataFormat::Invalid},
        .math_fidelity = math_fidelity_lofi, // no fidelity phases
        .result        = laid_out(src.shape.sizes, copy_dest_format(src.format, mods, settings), Storage::Dest),
    };
}

/// work: one output tile per src tile; no K.
constexpr Work copy_work(const Spec& src, const Spec&, const OpConfig&)
{
    return tile_grid_work(src);
}

/// plan: the unpack walk of one copy: one step, one run, every tile of src streamed in L1 order, each one tile further
/// on. Tile i of src lands in tile i of acc.
constexpr Plan copy_plan(const Spec& src, const Spec&, const copy_modifiers& mods, const OpConfig& config, const Work& work)
{
    constexpr std::uint32_t next_tile = 1; // tiles from src tile i to tile i + 1
    const UnpackPlan src_plan {
        .to              = config.in0.where,
        .reads           = src.format,
        .writes          = config.in0.format,
        .face_r_dim      = work.face_r_dim,
        .num_faces       = work.num_faces,
        .tile_bytes      = tile_bytes(src),
        .layout          = Layout::Tilized,
        .broadcast       = Broadcast::None,
        .transpose_faces = mods.transpose,
        .per_tile        = next_tile,
        .per_run         = 0, // one run
        .per_step        = 0, // one step
    };
    const UnpackWalk walk {.steps = 1, .runs = 1, .tiles_per_run = work.tile_rows * work.tile_cols, .held = Where::None};
    const MathPlan math {
        .math_fidelity   = config.math_fidelity,
        .dest_fp32       = is_32bit_format(config.result.format),
        .format          = config.in0.format,
        .transpose_faces = mods.transpose,
    };
    return {
        .in0  = src_plan,
        .in1  = {}, // to = Where::None: unpacker 1 idles
        .walk = walk,
        .math = math,
        .pack = {},
        .work = work,
    };
}

inline constexpr OpApi<copy_modifiers> copy_op {&copy_accepts, &copy_options, &copy_work, &copy_plan};

} // namespace llk2
