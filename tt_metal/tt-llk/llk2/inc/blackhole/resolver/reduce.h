// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reduce's OpApi on Blackhole: a row reduction (max or sum over each row's columns) of an [R, N] grid of tiles into
// an [R, 1] grid, one value per row in column 0 of the output tile, as LLK 1.0's reduce<*, REDUCE_ROW> and
// reduce_block_max_row leave it. in0 is the data, in1 the scaler tile (SDPA's identity_scale CB).
//
//   accepts   in in L1 in 32x32 tiles of an 8-bit-exponent float format, contiguous or a window of a wider tile grid;
//             a one-tile scaler of the same kind; a row reduce (max or sum)
//   options   in to SrcA, the scaler to SrcB; one fidelity phase for max, the kernel's fidelity for sum; the result
//             one tile per tile row of in, in the kernel's Dest format
//   work      tile_rows output tiles, each folding k_tiles = N input tiles
//   plan      the unpack walk (below), and math running the kind, into acc or folded into what it holds
//
// What the unpackers do (LLK 1.0's _llk_unpack_AB_reduce_block_max_row_): in0 to SrcA, each 16x16 face transposed on
// the way in (Haloize_mode), one whole tile per UNPACR; the scaler to SrcB, unpacked once per output tile and held
// while the N tiles of its tile row stream past (UnpackWalk::held = SrcB).
//   walk: 1 step; one run per output tile (R runs); N streamed tiles per run
//   in0:  next tile +1, next run + its tile-row step: N, or more when in0 is a window of a wider tile grid (the first
//         N columns of SDPA's scores in the narrow last K chunk)
//   in1:  never moves: the same scaler tile for every run
// Math (llk_lib/reduce.h) pools the transposed faces into one Dest row per face row of the tile, then transposes
// those rows into column 0.
//
// The scaler: GMPOOL scales each datum by the exponent of row 0 of the scaler's face 0, GAPOOL multiplies by it, so
// row 0 of face 0 is 1.0 for a plain max or sum (1/N for a mean). It must have the data's exponent width.
//
// Fidelity: max has no multiply (one phase). Sum runs the kernel's fidelity phases (GAPOOL multiplies by the
// scaler); with HiFi2 or more a BF16 sum with a 1.0 scaler is exact per product.
//
// A 32-bit Dest. A max: the result is FP32 with 19 bits kept (TF32: what a Src register holds), the rest 0, which is
// every bit of a max: its operands are Src datums. A sum's pooled rows would be cut to those 19 bits on their way to
// column 0 (llk_lib/reduce.h has why), so there the sum is a product instead (reduce_sums_by_product): in times the
// scaler's row 0 as a column, which MVMUL adds up in Dest at full width. Its plan is matmul's (resolver/matmul.h) for
// in [R, N] and the scaler tile transposed, used for every K step: in to SrcB, the scaler to SrcA. The scaler's row 0
// must then be 1.0 (or the mean's 1/N) over all 32 columns, not only face 0's 16. Transposed, a reduce scaler has that
// row in column 0 and face 2's row 0 in column 16: the product has the sums in both, and math clears the faces of
// column 16 (llk_lib/reduce.h).
//
// Not ported: column and scalar reduce, the 5-bit-exponent (FP16) formats, tiles other than 32x32.

#pragma once

#include <cstdint>

#include "resolver/formats.h"
#include "resolver/matmul.h" // matmul_plan: the walk of a sum in a 32-bit Dest
#include "resolver/modifiers.h"
#include "resolver/resolver.h"

namespace llk2
{

/// A sum in a 32-bit Dest is computed as a product with the scaler (the file comment); every other reduce by pooling.
constexpr bool reduce_sums_by_product(const reduce_modifiers& mods, const KernelSettings& settings)
{
    return settings.dest_fp32 && mods.kind == ReduceKind::Sum;
}

/// accepts: see the file comment.
constexpr Accepts reduce_accepts(const Spec& in, const Spec& scaler, const reduce_modifiers& mods, const KernelSettings&)
{
    const bool storage   = in.storage == Storage::L1 && scaler.storage == Storage::L1;
    const bool tiles     = has_default_tiles(in.shape) && has_default_tiles(scaler.shape);
    const bool shapes    = scaler.shape.sizes[TileY] == 1 && scaler.shape.sizes[TileX] == 1; // one scaler tile
    const bool layout    = has_dense_tile_rows(in.shape) && is_contiguous(scaler.shape);     // in may be a window: its rows step by tile_row_step
    const bool formats   = has_8bit_exponent(in.format) && has_8bit_exponent(scaler.format); // 5-bit exponents: not ported
    const bool modifiers = mods.dim == ReduceDim::Row && (mods.kind == ReduceKind::Max || mods.kind == ReduceKind::Sum);
    // A run-time size (resolver/work_counts.h): in's tile columns, the tiles folded into each output tile.
    const bool runtime = (!has_runtime_size(in) || is_runtime_size(in, TileX)) && !has_runtime_size(scaler);
    return {.storage = storage, .tiles = tiles, .shapes = shapes, .layout = layout, .formats = formats, .modifiers = modifiers, .runtime = runtime};
}

/// options: in to SrcA, the scaler to SrcB (the other way round for a sum by product); the result is one tile per
/// tile row of in, in the kernel's Dest format.
constexpr OpConfig reduce_options(const Spec& in, const Spec& scaler, const reduce_modifiers& mods, const KernelSettings& settings)
{
    Dims out              = in.shape.sizes;
    out[TileX]            = 1; // one output tile per tile row
    const bool multiplies = mods.kind == ReduceKind::Sum;
    const bool by_product = reduce_sums_by_product(mods, settings);
    return {
        .in0           = {.where = by_product ? Where::SrcB : Where::SrcA, .format = src_register_format(in.format, settings.dest_fp32)},
        .in1           = {.where = by_product ? Where::SrcA : Where::SrcB, .format = src_register_format(scaler.format, settings.dest_fp32)},
        .math_fidelity = multiplies ? settings.math_fidelity : math_fidelity_lofi, // max has no multiply: one phase
        .result        = laid_out(out, dest_format_for(settings), Storage::Dest),  // GMPOOL / GAPOOL write B formats as BF16 or FP32
    };
}

/// work: tile_rows output tiles, each folding k_tiles = N input tiles (in's tile columns).
constexpr Work reduce_work(const Spec& in, const Spec&, const OpConfig& config)
{
    const Dims& out = config.result.shape.sizes;
    return {
        .tile_rows  = out[TileY],
        .tile_cols  = out[TileX],
        .k_tiles    = in.shape.sizes[TileX],
        .face_r_dim = in.shape.sizes[DatumY],
        .num_faces  = faces_per_tile(in.shape),
        .k_from     = in0_count_source(in, TileX), // a run-time row length
    };
}

/// plan of a sum by product: matmul's for in [R, N] times the scaler transposed, the scaler's one tile at every K
/// step and under every output tile.
constexpr Plan reduce_product_plan(const Spec& in, const Spec& scaler, const reduce_modifiers& mods, const OpConfig& config, const Work& work)
{
    Plan plan            = matmul_plan(in, scaler, matmul_modifiers {.transpose_in1 = true}, config, work);
    plan.in1.per_tile    = 0;
    plan.in1.per_run     = 0;
    plan.in1.per_step    = 0;
    plan.math.reduce     = mods.kind;
    plan.math.accumulate = mods.accumulate;
    plan.math.by_product = true;
    return plan;
}

/// plan: the unpack walk described at the top of this file, and math running the kind (and seed) of mods.
constexpr Plan reduce_plan(const Spec& in, const Spec& scaler, const reduce_modifiers& mods, const OpConfig& config, const Work& work)
{
    if (config.in0.where == Where::SrcB) // a sum by product (reduce_options)
    {
        return reduce_product_plan(in, scaler, mods, config, work);
    }
    const std::uint32_t next_tile     = 1;                       // tiles from (r, n) to (r, n + 1)
    const std::uint32_t next_tile_row = tile_row_step(in.shape); // tiles from (r, 0) to (r + 1, 0): k_tiles, or more for a window
    const std::uint32_t no_step       = 0;                       // the walk has one step

    const UnpackPlan in_plan {
        .to              = Where::SrcA,
        .reads           = in.format,
        .writes          = config.in0.format,
        .face_r_dim      = work.face_r_dim,
        .num_faces       = work.num_faces,
        .tile_bytes      = tile_bytes(in),
        .layout          = Layout::Tilized,
        .broadcast       = Broadcast::None,
        .transpose_faces = true, // inside each face: GMPOOL / GAPOOL fold SrcA's columns, i.e. in's rows
        .per_tile        = next_tile,
        .per_run         = next_tile_row,
        .per_step        = no_step,
    };
    const UnpackPlan scaler_plan {
        .to              = Where::SrcB,
        .reads           = scaler.format,
        .writes          = config.in1.format,
        .face_r_dim      = scaler.shape.sizes[DatumY],
        .num_faces       = faces_per_tile(scaler.shape),
        .tile_bytes      = tile_bytes(scaler),
        .layout          = Layout::Tilized,
        .broadcast       = Broadcast::None,
        .transpose_faces = false,
        .per_tile        = 0, // held, unpacked once per run
        .per_run         = 0, // the same scaler for every output tile
        .per_step        = no_step,
    };
    const UnpackWalk walk {
        .steps         = 1,
        .runs          = work.tile_rows,
        .tiles_per_run = work.k_tiles,
        .held          = Where::SrcB,
        .steps_of      = WorkCount::None,
        .runs_of       = WorkCount::Rows,
        .tiles_of      = WorkCount::K,
    };
    const MathPlan math {
        .math_fidelity   = config.math_fidelity,
        .dest_fp32       = is_32bit_format(config.result.format),
        .format          = config.in0.format,
        .transpose_faces = false, // math transposes the pooled rows itself
        .reduce          = mods.kind,
        .accumulate      = mods.accumulate,
    };
    return {.in0 = in_plan, .in1 = scaler_plan, .walk = walk, .math = math, .pack = {}, .work = work};
}

inline constexpr OpApi<reduce_modifiers> reduce_op {&reduce_accepts, &reduce_options, &reduce_work, &reduce_plan};

} // namespace llk2
