// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The eltwise binary OpApis on Blackhole: add, sub and mul. in0 unpacks to SrcA, in1 to SrcB, as tt-llk does; the
// FPU writes in0 <function> in1 into Dest, one 8-row block per instruction (ELWADD, ELWSUB, ELWMUL).
//
//   accepts   in0 and in1 in L1 in 32x32 float tiles, each contiguous or a window of a wider tile grid, of one
//             exponent width; in1 shaped as the broadcast needs (the table below)
//   options   in0 to SrcA, in1 to SrcB; one fidelity phase for add and sub, the kernel's fidelity for mul; the
//             result in Dest with in0's shape
//   work      one output tile per in0 tile
//   plan      the unpack walk (below), and math running the op's BinaryFunction
//
// The three ops are one family: they differ only in BinaryFunction, which the Plan carries to llk-lib's math; accepts
// and work are shared. options differ in the fidelity: add and sub run one phase, mul runs the kernel's (ELWMUL
// multiplies only a few mantissa bits per phase; llk-lib's math loops over the phases). mul alone takes .accumulate
// (y = acc + in0 * in1): ELWMUL always adds to Dest, so the Plan only tells math not to zero the output tiles first.
// The three share one modifier struct (eltwise_binary_modifiers), so each step is one function, a template over the
// BinaryFunction only where the function matters.
//
// in1's shape per Broadcast, for in0 of [rt, ct] tiles:
//   None    [rt, ct]  tile (r, c) pairs with in0's tile (r, c)
//   Row     [1, ct]   a bias: row 0 of tile c spread down every row of in0's tile column c
//   Col     [rt, 1]   column 0 of tile r spread across every column of in0's tile row r (SDPA's per-row statistics:
//                     a row max, a correction factor or 1/sum, stored in column 0)
//   Scalar  [1, 1]    datum [0, 0] spread over every datum
//
// Unpack walk: every operand streams, nothing is held. A run is one row of output tiles: in0 moves one tile per
// streamed tile and one tile row per run. in1 moves the same way when it has in0's shape; a broadcast in1 stays on
// its tile where it has only one (Row: one tile row, so back to its first tile every run; Col: one tile per row, so
// it stays inside a run; Scalar: one tile). The unpacker reads only the faces the broadcast reads (unpack.h).
// A tile row is the operand's tile-row step (tile_row_step): its own width when it is contiguous, its parent's when
// it is a window of a wider tile grid (llk2::cols of a tensor with several tile rows), as in matmul and reduce.
// (1.0's SDPA sub holds a column-broadcast in1 in SrcB for the run instead, llk_unpack_AB_sub_bcast_col_custom: the
// same result with tile_cols - 1 fewer tiles through unpacker 1 per run.)
//
// Run-time sizes (resolver/work_counts.h). in0 may mark a tile-grid size run time: its tile rows (a run-time number of
// runs; per_run is a stride, compile time), or the tile columns of one tile row (a run-time number of streamed
// tiles). in1 marks the same size, unless the broadcast leaves it one tile row or column there (SDPA: a block of a
// row of S, a run-time number of tiles, minus the row's one m tile with Broadcast::Col). work names the count's
// source and plan names the walk's loops after the Work's counts; nothing else in the Plan depends on the count.
//
// Not ported: integer formats, tiles other than 32x32, .accumulate for add and sub (ELWADD and ELWSUB can add to
// Dest; llk-lib issues them as overwrite).

#pragma once

#include <cstdint>

#include "resolver/formats.h"
#include "resolver/modifiers.h"
#include "resolver/resolver.h"

namespace llk2
{

/// The Dest format an eltwise binary writes: FP32 with a 32-bit Dest, else FP16 or BF16 by the Src style (ELWADD.md).
constexpr DataFormat eltwise_binary_dest_format(DataFormat in, bool dest_fp32)
{
    if (dest_fp32)
    {
        return DataFormat::Float32;
    }
    return has_5bit_exponent(in) ? DataFormat::Float16 : DataFormat::Float16_b;
}

/// in1 is one row of tiles spread over every row of in0's tiles.
constexpr bool eltwise_binary_is_bias_row(const Spec& in0, const Spec& in1)
{
    return in1.shape.sizes[TileY] == 1 && in1.shape.sizes[TileX] == in0.shape.sizes[TileX];
}

/// in1 is one column of tiles spread over every column of in0's tiles.
constexpr bool is_eltwise_binary_bias_col(const Spec& in0, const Spec& in1)
{
    return in1.shape.sizes[TileY] == in0.shape.sizes[TileY] && in1.shape.sizes[TileX] == 1;
}

/// in1 has the shape Broadcast b needs for in0 (the table at the top of this file).
constexpr bool eltwise_binary_is_in1_shape(const Spec& in0, const Spec& in1, Broadcast b)
{
    switch (b)
    {
        case Broadcast::None:
            return equal(in0.shape.sizes, in1.shape.sizes);
        case Broadcast::Row:
            return eltwise_binary_is_bias_row(in0, in1);
        case Broadcast::Col:
            return is_eltwise_binary_bias_col(in0, in1);
        case Broadcast::Scalar:
            return in1.shape.sizes[TileY] == 1 && in1.shape.sizes[TileX] == 1;
    }
    return false;
}

/// Tiles in1's address moves from one of in0's tiles to the next, inside a run (one output tile row): in1 has one tile
/// per output tile column only without a broadcast or with a row one.
constexpr std::uint32_t eltwise_binary_in1_per_tile(Broadcast b)
{
    return b == Broadcast::None || b == Broadcast::Row ? 1 : 0;
}

/// Tiles in1's address moves from one run (output tile row) to the next: in1's tile-row step when it has a tile row
/// per run (no broadcast: a whole row of its tiles; a column broadcast: one tile; more when in1 is a window), nothing
/// when in1 has a single tile row.
constexpr std::uint32_t eltwise_binary_in1_per_run(Broadcast b, const Spec& in1)
{
    return b == Broadcast::None || b == Broadcast::Col ? tile_row_step(in1.shape) : 0;
}

/// accepts: see the file comment. F is the op's BinaryFunction.
template <BinaryFunction F>
constexpr Accepts eltwise_binary_accepts(const Spec& in0, const Spec& in1, const eltwise_binary_modifiers& mods, const KernelSettings& settings)
{
    const bool storage = in0.storage == Storage::L1 && in1.storage == Storage::L1;
    const bool tiles   = has_default_tiles(in0.shape) && has_default_tiles(in1.shape);
    const bool shapes  = eltwise_binary_is_in1_shape(in0, in1, mods.bcast);                // the table at the top of this file
    const bool layout  = has_dense_tile_rows(in0.shape) && has_dense_tile_rows(in1.shape); // either may be a window: plan reads tile_row_step
    // Both operands are read in SrcA's style (ELWADD.md): same exponent width. Integers: not ported.
    const bool formats = src_register_format(in0.format, settings.dest_fp32) != DataFormat::Invalid &&
                         src_register_format(in1.format, settings.dest_fp32) != DataFormat::Invalid && !is_integer_format(in0.format) &&
                         !is_integer_format(in1.format) && has_5bit_exponent(in0.format) == has_5bit_exponent(in1.format);
    const bool modifiers = !mods.accumulate || F == BinaryFunction::Mul; // accumulate for add and sub: not ported
    // A run-time size: in0's tile rows, or the tile columns of one tile row (the output tiles follow each other in
    // Dest, so a short row of several would leave gaps). in1 marks the same size where it has that dim, and none
    // where the broadcast gives it a single tile row or column there.
    const bool in0_runtime     = !has_runtime_size(in0) || is_runtime_size(in0, TileY) || in0.shape.sizes[TileY] == 1;
    const bool in1_has_rows    = mods.bcast == Broadcast::None || mods.bcast == Broadcast::Col;
    const bool in1_has_columns = mods.bcast == Broadcast::None || mods.bcast == Broadcast::Row;
    const bool in1_has_dim     = is_runtime_size(in0, TileY) ? in1_has_rows : in1_has_columns;
    const bool in1_follows     = in1_has_dim ? same_runtime_size(in0, in1) : !has_runtime_size(in1);
    const bool runtime         = in0_runtime && in1_follows;
    return {.storage = storage, .tiles = tiles, .shapes = shapes, .layout = layout, .formats = formats, .modifiers = modifiers, .runtime = runtime};
}

/// options: in0 to SrcA, in1 to SrcB, the result in Dest with in0's shape. ELWADD and ELWSUB run one fidelity phase
/// whatever the kernel's fidelity is; ELWMUL runs the kernel's (F is the op's BinaryFunction).
template <BinaryFunction F>
constexpr OpConfig eltwise_binary_options(const Spec& in0, const Spec& in1, const eltwise_binary_modifiers&, const KernelSettings& settings)
{
    return {
        .in0           = {.where = Where::SrcA, .format = src_register_format(in0.format, settings.dest_fp32)},
        .in1           = {.where = Where::SrcB, .format = src_register_format(in1.format, settings.dest_fp32)},
        .math_fidelity = F == BinaryFunction::Mul ? settings.math_fidelity : math_fidelity_lofi,
        .result        = laid_out(in0.shape.sizes, eltwise_binary_dest_format(in0.format, settings.dest_fp32), Storage::Dest),
    };
}

/// work: one output tile per in0 tile (one input tile pair per output tile); no K. A size in0 marks run time is a
/// run-time count, from in0.
constexpr Work eltwise_binary_work(const Spec& in0, const Spec&, const OpConfig&)
{
    return in0_tile_grid_work(in0);
}

/// plan: the unpack walk described at the top of this file, and math running F (into acc with mul's .accumulate).
template <BinaryFunction F>
constexpr Plan eltwise_binary_plan(const Spec& in0, const Spec& in1, const eltwise_binary_modifiers& mods, const OpConfig& config, const Work& work)
{
    const std::uint32_t next_tile     = 1;                        // in0's tiles from (r, c) to (r, c + 1)
    const std::uint32_t next_tile_row = tile_row_step(in0.shape); // in0's tiles from (r, c) to (r + 1, c): tile_cols, or more for a window
    const std::uint32_t no_step       = 0;                        // the walk has one step

    const UnpackPlan in0_plan {
        .to              = Where::SrcA,
        .reads           = in0.format,
        .writes          = config.in0.format,
        .face_r_dim      = work.face_r_dim,
        .num_faces       = work.num_faces,
        .tile_bytes      = tile_bytes(in0),
        .layout          = Layout::Tilized,
        .broadcast       = Broadcast::None,
        .transpose_faces = false,
        .per_tile        = next_tile,
        .per_run         = next_tile_row,
        .per_step        = no_step,
    };
    const UnpackPlan in1_plan {
        .to              = Where::SrcB,
        .reads           = in1.format,
        .writes          = config.in1.format,
        .face_r_dim      = in1.shape.sizes[DatumY],
        .num_faces       = faces_per_tile(in1.shape),
        .tile_bytes      = tile_bytes(in1),
        .layout          = Layout::Tilized,
        .broadcast       = mods.bcast,
        .transpose_faces = false,
        .per_tile        = eltwise_binary_in1_per_tile(mods.bcast),     // along an output tile row
        .per_run         = eltwise_binary_in1_per_run(mods.bcast, in1), // to the next output tile row
        .per_step        = no_step,
    };
    const UnpackWalk walk {
        .steps         = 1,
        .runs          = work.tile_rows,
        .tiles_per_run = work.tile_cols,
        .held          = Where::None,
        .steps_of      = WorkCount::None,
        .runs_of       = WorkCount::Rows,
        .tiles_of      = WorkCount::Cols,
    };
    const MathPlan math {
        .math_fidelity   = config.math_fidelity,
        .dest_fp32       = is_32bit_format(config.result.format),
        .format          = config.in0.format,
        .transpose_faces = false,
        .binary          = F,
        .accumulate      = mods.accumulate,
    };
    return {.in0 = in0_plan, .in1 = in1_plan, .walk = walk, .math = math, .pack = {}, .work = work};
}

/// in0 + in1.
inline constexpr OpApi<eltwise_binary_modifiers> add_op {
    &eltwise_binary_accepts<BinaryFunction::Add>,
    &eltwise_binary_options<BinaryFunction::Add>,
    &eltwise_binary_work,
    &eltwise_binary_plan<BinaryFunction::Add>,
};

/// in0 - in1.
inline constexpr OpApi<eltwise_binary_modifiers> sub_op {
    &eltwise_binary_accepts<BinaryFunction::Sub>,
    &eltwise_binary_options<BinaryFunction::Sub>,
    &eltwise_binary_work,
    &eltwise_binary_plan<BinaryFunction::Sub>,
};

/// in0 * in1, in the kernel's fidelity; with .accumulate, acc + in0 * in1.
inline constexpr OpApi<eltwise_binary_modifiers> mul_op {
    &eltwise_binary_accepts<BinaryFunction::Mul>,
    &eltwise_binary_options<BinaryFunction::Mul>,
    &eltwise_binary_work,
    &eltwise_binary_plan<BinaryFunction::Mul>,
};

} // namespace llk2
