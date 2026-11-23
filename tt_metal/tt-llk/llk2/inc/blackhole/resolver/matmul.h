// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Matmul's OpApi on Blackhole. in0 [M, K] unpacks to SrcB, in1 [K, N] to SrcA, as tt-llk does; the FPU accumulates
// tile_rows x tile_cols output tiles in Dest over k_tiles K steps.
//
//   accepts   in0 [M, K] and in1 [K, N] in L1 in 32x32 tiles, each contiguous or a window of a wider tile grid,
//             formats the unpacker can write to Src, both float or both integer (integers need the 32-bit Dest),
//             adding into acc (.accumulate)
//   options   in0 to SrcB, in1 to SrcA, the kernel's fidelity, the [M, N] result in Dest
//   work      the output tile grid, K tiles, in0's tile geometry
//   plan      the unpack walk (below) and math on in1's Src format
//
// Unpack walk. A K step multiplies in0's tile column k (tile_rows tiles) by in1's tile row k (tile_cols tiles). The
// longer output dimension is reused: when tile_cols >= tile_rows, each in0 tile is held in SrcB for one run while the
// tile_cols in1 tiles of row k stream through SrcA; otherwise each in1 tile is held in SrcA while the tile_rows in0
// tiles of column k stream through SrcB. in0 is [tile_rows, k_tiles] tiles and in1 [k_tiles, tile_cols], both
// row-major, so moving down in0's column is k_tiles tiles and moving along in1's row is 1 tile. Either may be a window
// of a wider tile grid (llk2::cols of a wider tensor): moving down a column is then the parent's tile-row step
// (tile_row_step).
//
// transpose_in1 uses each in1 tile transposed (the unpacker transposes inside each face, math swaps faces 1 and 2),
// as tt-llk's matmul with transpose.
//
// Run-time sizes (resolver/work_counts.h). K may be a run-time count: in0 marks its tile columns and in1 its tile rows.
// Or N, the output's tile columns: in1 marks its tile columns, and in0 is one tile row (SDPA: a Q row times a block of
// K^T of a run-time number of tiles). work names the count's source and plan names the walk's loops after the Work's
// counts. No address step depends on either: they are strides, which stay compile time (moving down in0's column or
// in1's column is the tensor's tile-row step, a view's parent's).
//
// Not ported: tiles other than 32x32; a matmul that overwrites acc (.accumulate = false: MVMUL adds into Dest, so it
// would zero acc's tiles first, a ZEROACC per output tile that the ring's zeroed sections make unnecessary).

#pragma once

#include <cstdint>

#include "resolver/formats.h"
#include "resolver/modifiers.h"
#include "resolver/resolver.h"

namespace llk2
{

/// Sizes of the product [M, N] of in0 [M, K] and in1 [K, N]: the tile rows and the tile's height from in0's sizes a,
/// the tile columns and the tile's width from in1's sizes b.
constexpr Dims matmul_product_sizes(const Dims& a, const Dims& b)
{
    return {a[TileY], b[TileX], a[FaceY], b[FaceX], a[DatumY], b[DatumX]};
}

/// The Dest format matmul accumulates into: Int32 for integer inputs (needs the 32-bit Dest mode), else FP32 or BF16.
constexpr DataFormat matmul_dest_format(DataFormat in, bool dest_fp32)
{
    if (is_integer_format(in))
    {
        return DataFormat::Int32;
    }
    return dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
}

/// accepts: see the file comment.
constexpr Accepts matmul_accepts(const Spec& in0, const Spec& in1, const matmul_modifiers& mods, const KernelSettings& settings)
{
    const bool storage = in0.storage == Storage::L1 && in1.storage == Storage::L1;
    const bool tiles   = has_default_tiles(in0.shape) && has_default_tiles(in1.shape);
    const bool shapes  = in0.shape.sizes[TileX] == in1.shape.sizes[TileY]; // K matches
    const bool layout  = has_dense_tile_rows(in0.shape) && has_dense_tile_rows(in1.shape); // either may be a window: plan reads tile_row_step
    const bool formats = src_register_format(in0.format, settings.dest_fp32) != DataFormat::Invalid &&
                         src_register_format(in1.format, settings.dest_fp32) != DataFormat::Invalid &&
                         is_integer_format(in0.format) == is_integer_format(in1.format);
    const bool modifiers = mods.accumulate;                                      // a matmul that overwrites acc: not ported
    const bool dest_mode = !is_integer_format(in0.format) || settings.dest_fp32; // integers accumulate in a 32-bit Dest
    // A run-time size: K (in0's tile columns and in1's tile rows, both marked), or the output's tile columns N (in1's
    // tile columns) when in0 is one tile row: the output tiles follow each other in Dest.
    const bool runtime_k = is_runtime_size(in0, TileX) && is_runtime_size(in1, TileY);
    const bool runtime_n = !has_runtime_size(in0) && is_runtime_size(in1, TileX) && in0.shape.sizes[TileY] == 1;
    const bool runtime   = (!has_runtime_size(in0) && !has_runtime_size(in1)) || runtime_k || runtime_n;
    return {
        .storage   = storage,
        .tiles     = tiles,
        .shapes    = shapes,
        .layout    = layout,
        .formats   = formats,
        .modifiers = modifiers,
        .dest_mode = dest_mode,
        .runtime   = runtime};
}

/// options: in0 to SrcB, in1 to SrcA (as tt-llk), the kernel's fidelity, the [M, N] result in Dest.
constexpr OpConfig matmul_options(const Spec& in0, const Spec& in1, const matmul_modifiers&, const KernelSettings& settings)
{
    return {
        .in0           = {.where = Where::SrcB, .format = src_register_format(in0.format, settings.dest_fp32)},
        .in1           = {.where = Where::SrcA, .format = src_register_format(in1.format, settings.dest_fp32)},
        .math_fidelity = settings.math_fidelity,
        .result        = laid_out(matmul_product_sizes(in0.shape.sizes, in1.shape.sizes), matmul_dest_format(in0.format, settings.dest_fp32), Storage::Dest),
    };
}

/// work: the output tile grid, K tiles, and in0's tile geometry.
constexpr Work matmul_work(const Spec& in0, const Spec& in1, const OpConfig& config)
{
    const Dims& out = config.result.shape.sizes;
    return {
        .tile_rows  = out[TileY],
        .tile_cols  = out[TileX],
        .k_tiles    = in0.shape.sizes[TileX],
        .face_r_dim = in0.shape.sizes[DatumY],
        .num_faces  = faces_per_tile(in0.shape),
        .cols_from  = in1_count_source(in1, TileX), // a run-time N
        .k_from     = in0_count_source(in0, TileX), // a run-time K
    };
}

/// Whether in0 is the held (reused) operand: see "Unpack walk" in the file comment.
constexpr bool matmul_reuses_in0(const Work& work)
{
    return work.tile_cols >= work.tile_rows;
}

/// plan: the unpack walk of the file comment, and math on in1's Src format.
constexpr Plan matmul_plan(const Spec& in0, const Spec& in1, const matmul_modifiers& mods, const OpConfig& config, const Work& work)
{
    const bool reuse_in0          = matmul_reuses_in0(work);
    const std::uint32_t in0_down  = tile_row_step(in0.shape); // tiles from in0(r, k) to in0(r + 1, k): k_tiles, or more for a window
    const std::uint32_t in0_right = 1;                        // tiles from in0(r, k) to in0(r, k + 1)
    const std::uint32_t in1_right = 1;                        // tiles from in1(k, c) to in1(k, c + 1)
    const std::uint32_t in1_down  = tile_row_step(in1.shape); // tiles from in1(k, c) to in1(k + 1, c): tile_cols, or more for a window

    const UnpackPlan in0_plan {
        .to              = Where::SrcB,
        .reads           = in0.format,
        .writes          = config.in0.format,
        .face_r_dim      = work.face_r_dim,
        .num_faces       = work.num_faces,
        .tile_bytes      = tile_bytes(in0),
        .layout          = Layout::Tilized,
        .broadcast       = Broadcast::None,
        .transpose_faces = false,
        .per_tile        = reuse_in0 ? 0 : in0_down, // streams down its column when in1 is the held one
        .per_run         = reuse_in0 ? in0_down : 0, // next held tile, one row down
        .per_step        = in0_right,                // next K column
    };
    const UnpackPlan in1_plan {
        .to              = Where::SrcA,
        .reads           = in1.format,
        .writes          = config.in1.format,
        .face_r_dim      = in1.shape.sizes[DatumY],
        .num_faces       = faces_per_tile(in1.shape),
        .tile_bytes      = tile_bytes(in1),
        .layout          = Layout::Tilized,
        .broadcast       = Broadcast::None,
        .transpose_faces = mods.transpose_in1,
        .per_tile        = reuse_in0 ? in1_right : 0, // streams along its row when in0 is the held one
        .per_run         = reuse_in0 ? 0 : in1_right, // next held tile, one column right
        .per_step        = in1_down,                  // next K row
    };
    const UnpackWalk walk {
        .steps         = work.k_tiles,
        .runs          = reuse_in0 ? work.tile_rows : work.tile_cols, // one per held tile
        .tiles_per_run = reuse_in0 ? work.tile_cols : work.tile_rows, // the streamed tiles
        .held          = reuse_in0 ? Where::SrcB : Where::SrcA,
        .steps_of      = WorkCount::K,
        .runs_of       = reuse_in0 ? WorkCount::Rows : WorkCount::Cols,
        .tiles_of      = reuse_in0 ? WorkCount::Cols : WorkCount::Rows,
    };
    const MathPlan math {
        .math_fidelity   = config.math_fidelity,
        .dest_fp32       = is_32bit_format(config.result.format),
        .format          = config.in1.format,
        .transpose_faces = mods.transpose_in1,
        .accumulate      = mods.accumulate,
    };
    return {.in0 = in0_plan, .in1 = in1_plan, .walk = walk, .math = math, .pack = {}, .work = work};
}

inline constexpr OpApi<matmul_modifiers> matmul_op {&matmul_accepts, &matmul_options, &matmul_work, &matmul_plan};

} // namespace llk2
