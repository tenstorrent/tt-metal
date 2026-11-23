// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the Resolver returns for one Compute API call (tt-llk-lab llk-tensor, report 3, section 3:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor). Three results, of the
// three Resolver steps that follow accepts (which returns only a bool):
//   OpConfig  options (negotiation): where math reads each operand and in which register format, the fidelity,
//             and the spec of the result
//   Work      work (allocation): loop counts, from the tensors once the config is fixed
//   Plan      plan: what each unit does. llk-lib turns it into each thread's parameter struct
//             (unpack_params_from, matmul_params_from, ...)
//
// The Plan describes hardware work, not the op: llk-lib's unpack, for example, reads only the operands' streams and
// the walk below, so one unpack implementation serves every op. Op-specific knowledge lives in each op's plan()
// (resolver/<op>.h, per arch).

#pragma once

#include <cstdint>

#include "resolver/reduce_function.h" // ReduceKind
#include "resolver/vec_function.h"    // VecFunction
#include "tensor/tensor.h"

namespace llk2
{

/// Where an operand goes, or where math reads it from.
enum class Where : std::uint8_t
{
    None, ///< nowhere: the operand is not unpacked
    SrcA, ///< SrcA (unpacker 0)
    SrcB, ///< SrcB (unpacker 1)
    Dest  ///< Dest (unpacker 0 straight to Dest, or an operand already in Dest)
};

/// How an operand's tile is read when it is smaller than the other operand along an axis.
enum class Broadcast : std::uint8_t
{
    None,  ///< no broadcast: the operands have the same shape
    Row,   ///< one row repeated down every row
    Col,   ///< one column repeated across every column
    Scalar ///< one datum everywhere
};

/// A modifier that is one config bit: fixed off, fixed on, or chosen at run time by an argument of the call
/// (resolver/modifiers.h has the rules). Defined here, beside Broadcast, because Plans carry it.
enum class Toggle : std::uint8_t
{
    Off,    ///< always off: no run-time argument, the bit is written as an immediate
    On,     ///< always on: no run-time argument, the bit is written as an immediate
    Runtime ///< the call takes the value as an argument; compile-time checks treat the feature as possibly on
};

/// How a tensor sits in L1.
enum class Layout : std::uint8_t
{
    Tilized, ///< tiles, each tile face by face
    RowMajor ///< plain rows, tilized by the unpacker on the way in. Not ported: no Resolver plans it, llk-lib's unpack refuses it
};

/// No K: one input tile (or tile pair) per output tile. Work::k_tiles of every op but matmul and reduce.
inline constexpr std::uint32_t no_k_steps = 1;

/// An operand as math reads it: the register and the format in it.
struct OperandRep
{
    Where where;       ///< where math reads it
    DataFormat format; ///< the register format the unpacker writes (pack: the packer's input format)
};

/// Negotiation's result for one op.
struct OpConfig
{
    OperandRep in0;                 ///< the first operand
    OperandRep in1;                 ///< the second operand
    std::uint32_t math_fidelity;    ///< ckernel::MathFidelity value the op runs with
    Spec result;                    ///< what the op produces
};

/// Where the value of one of a Work's counts comes from: the Work itself (compile time), or one of the call's tensors
/// at run time (a size its spec marks run time, tensor/tensor.h; the Work then holds the bound).
enum class CountSource : std::uint8_t
{
    Fixed,    ///< compile time: the number in the Work
    In0TileY, ///< in0's tile rows
    In0TileX, ///< in0's tile columns
    In1TileY, ///< in1's tile rows
    In1TileX  ///< in1's tile columns
};

/// Work allocation's result for one op. The three counts are the only values of a resolved call that may be run time:
/// each is a number and where its run-time value comes from (CountSource::Fixed: the number is the value; otherwise it
/// is the upper bound, and llk-lib's execute takes the value as an argument: resolver/work_counts.h).
/// tile_rows, tile_cols and k_tiles are what LLK 1.0 calls rt_dim, ct_dim and kt_dim.
struct Work
{
    std::uint32_t tile_rows;  ///< output tiles along y
    std::uint32_t tile_cols;  ///< output tiles along x
    std::uint32_t k_tiles;    ///< input tiles folded into each output tile: matmul's K, reduce's row length; no_k_steps otherwise
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    CountSource rows_from = CountSource::Fixed; ///< where tile_rows' value comes from
    CountSource cols_from = CountSource::Fixed; ///< where tile_cols' value comes from
    CountSource k_from    = CountSource::Fixed; ///< where k_tiles' value comes from
};

/// Where a Work count that is in0's tile-grid size d (TileY or TileX) comes from: in0, if it marks that size run
/// time; else the count is Fixed.
constexpr CountSource in0_count_source(const Spec& in0, Dim d)
{
    if (!is_runtime_size(in0, d))
    {
        return CountSource::Fixed;
    }
    return d == TileY ? CountSource::In0TileY : CountSource::In0TileX;
}

/// As in0_count_source, for a Work count that is in1's size d.
constexpr CountSource in1_count_source(const Spec& in1, Dim d)
{
    if (!is_runtime_size(in1, d))
    {
        return CountSource::Fixed;
    }
    return d == TileY ? CountSource::In1TileY : CountSource::In1TileX;
}

/// Which count of the call's Work a loop of the Plan runs to. The Plan keeps the number too (the bound, and the value
/// when the Work's count is fixed); the name is what lets llk-lib take the run-time value (resolver/work_counts.h).
enum class WorkCount : std::uint8_t
{
    None,   ///< the loop is not a Work count: it runs to the Plan's own number
    Rows,   ///< Work::tile_rows
    Cols,   ///< Work::tile_cols
    K,      ///< Work::k_tiles
    Tiles   ///< Work::tile_rows * Work::tile_cols: every output tile
};

/// The Work of an op with one output tile per tile of t and no K: t's tile grid and tile geometry.
constexpr Work tile_grid_work(const Spec& t)
{
    return {
        .tile_rows  = t.shape.sizes[TileY],
        .tile_cols  = t.shape.sizes[TileX],
        .k_tiles    = no_k_steps,
        .face_r_dim = t.shape.sizes[DatumY],
        .num_faces  = faces_per_tile(t.shape),
    };
}

/// tile_grid_work(in0) for an op whose in0 may mark a tile-grid size run time: that count is then in0's at run time.
constexpr Work in0_tile_grid_work(const Spec& in0)
{
    Work work    = tile_grid_work(in0);
    work.rows_from = in0_count_source(in0, TileY);
    work.cols_from = in0_count_source(in0, TileX);
    return work;
}

/// One operand as an unpacker reads it: formats, geometry, and how its L1 address moves through the walk.
/// Address steps are in tiles of this operand.
struct UnpackPlan
{
    Where to;                 ///< SrcA (unpacker 0), SrcB (unpacker 1), Dest (unpacker 0, to Dest), or None: not unpacked
    DataFormat reads;         ///< the format in L1
    DataFormat writes;        ///< the register format
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tile_bytes; ///< one tile in L1, in bytes
    Layout layout;            ///< how the operand sits in L1
    Broadcast broadcast;      ///< how a smaller operand is spread over the other
    bool transpose_faces;     ///< transpose inside each face on the way in
    std::uint32_t per_tile;   ///< tiles the address moves after each streamed tile (inside a run)
    std::uint32_t per_run;    ///< tiles the address moves from one run to the next
    std::uint32_t per_step;   ///< tiles the address moves from one step to the next
};

/// The order the unpackers walk their operands, for every op:
///   for each of `steps` steps:                    (matmul: K steps)
///     for each of `runs` runs:                    (matmul: one per reused tile)
///       unpack the held operand once, if any      (matmul: the reused tile)
///       repeat tiles_per_run times:               (one MOP iteration each)
///         unpack every streamed operand, then move its address by per_tile
struct UnpackWalk
{
    std::uint32_t steps;         ///< outer loop count
    std::uint32_t runs;          ///< runs per step
    std::uint32_t tiles_per_run; ///< streamed tiles per run
    Where held;                  ///< the operand unpacked once per run (SrcA or SrcB), or None: every operand streams
    WorkCount steps_of = WorkCount::None; ///< the Work count `steps` is
    WorkCount runs_of  = WorkCount::None; ///< the Work count `runs` is
    WorkCount tiles_of = WorkCount::None; ///< the Work count `tiles_per_run` is
};

/// Which FPU elementwise function an eltwise binary Plan runs (MathPlan::binary).
enum class BinaryFunction : std::uint32_t
{
    None, ///< no function
    Add,  ///< in0 + in1
    Sub,  ///< in0 - in1
    Mul   ///< in0 * in1
};

/// 1.0 as an FP32 bit pattern: the scale of an unscaled exp.
inline constexpr std::uint32_t vec_fp32_one = 0x3f800000u;

/// What math (the FPU or the SFPU) does.
struct MathPlan
{
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value
    bool dest_fp32;              ///< 32-bit Dest mode
    DataFormat format;           ///< the math (Src) format
    bool transpose_faces;        ///< write the faces in transposed order
    BinaryFunction binary = BinaryFunction::None; ///< eltwise binary: add, sub or mul
    VecFunction vec       = VecFunction::None;    ///< the SFPU functions: which one
    VecForm vec_form      = VecForm::Exact;       ///< the SFPU functions: which of the function's algorithms
    VecRegion vec_region  = VecRegion::All;       ///< the SFPU functions: the datums of each tile it rewrites
    std::uint32_t scale   = vec_fp32_one;         ///< exp: compile-time input scale, an FP32 bit pattern (exp(scale x))
    ReduceKind reduce     = ReduceKind::Max;      ///< reduce: max or sum; read by reduce only
    bool by_product       = false;                ///< reduce: the sum as a product with the scaler (a 32-bit Dest), not by pooling
    bool accumulate       = false;                ///< matmul, mul, reduce: fold the result into what Dest holds instead of overwriting it
};

/// What the packer does.
struct PackPlan
{
    DataFormat writes;        ///< the format in L1
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tile_bytes; ///< one tile in L1, in bytes
    bool relu;                ///< clamp negatives to 0 on the way out
    Toggle l1_acc;            ///< add into what is in L1 instead of overwriting it (Runtime: the pack call's argument)
    bool untilize;            ///< write rows instead of tiles
    DataFormat in_format;     ///< the packer's input format (In_data_format): the format the packer converts from
    bool dest_fp32;           ///< Dest holds 32-bit datums
    bool sum_tiles = false;   ///< every Dest tile goes to the same L1 tile: the first as l1_acc says, the rest added into it
};

/// What every unit does for one Compute API call.
struct Plan
{
    UnpackPlan in0;  ///< the first operand's stream
    UnpackPlan in1;  ///< the second operand's stream
    UnpackWalk walk; ///< the loop nest over the streams
    MathPlan math;   ///< math's part
    PackPlan pack;   ///< the packer's part
    Work work;       ///< loop counts and tile geometry
};

} // namespace llk2
