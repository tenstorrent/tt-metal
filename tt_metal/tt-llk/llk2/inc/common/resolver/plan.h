// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the Resolver returns for one Compute API call (tt-llk-lab llk-tensor, report 3, section 3:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor). Three results, one
// per Resolver step:
//   OpConfig  options (negotiation): which unit does the math, the register format of each operand, the fidelity,
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

/// The Compute API op a Plan belongs to.
enum class Op : std::uint8_t
{
    None,          ///< no op
    Matmul,        ///< matmul
    EltwiseBinary, ///< add, sub
    Copy,          ///< copy
    Vec,           ///< the SFPU activations
    Pack           ///< pack
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
/// (compute_api/modifiers.h has the rules). Defined here, beside Broadcast, because Plans carry it.
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
    RowMajor ///< plain rows, tilized by the unpacker on the way in
};

/// No K: one input tile (or tile pair) per output tile. Work::kt_dim of every op but matmul.
inline constexpr std::uint32_t no_k_steps = 1;

/// An operand where math reads it.
struct Rep
{
    Where where;       ///< where math reads it
    DataFormat format; ///< the register format the unpacker writes (pack: the packer's input format)
};

/// Negotiation's result for one op.
struct OpConfig
{
    Exu unit;                       ///< the unit that does the op's work
    Rep in0;                        ///< the first operand
    Rep in1;                        ///< the second operand
    std::uint32_t math_fidelity;    ///< ckernel::MathFidelity value the op runs with
    TensorSpec<tensor_rank> result; ///< what the op produces
};

/// Work allocation's result for one op.
struct Work
{
    std::uint32_t rt_dim;     ///< output tiles along y
    std::uint32_t ct_dim;     ///< output tiles along x
    std::uint32_t kt_dim;     ///< input tiles folded into each output tile (matmul's K; no_k_steps otherwise)
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
};

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
    std::uint32_t tile_step;  ///< tiles the address moves after each streamed tile (inside a run)
    std::uint32_t run_step;   ///< tiles the address moves from one run to the next
    std::uint32_t step_step;  ///< tiles the address moves from one step to the next
};

/// The order the unpackers walk their operands, for every op:
///   for each of `steps` steps:                    (matmul: K steps)
///     for each of `runs` runs:                    (matmul: one per reused tile)
///       unpack the held operand once, if any      (matmul: the reused tile)
///       repeat tiles_per_run times:               (one MOP iteration each)
///         unpack every streamed operand, then move its address by tile_step
struct UnpackWalk
{
    std::uint32_t steps;         ///< outer loop count
    std::uint32_t runs;          ///< runs per step
    std::uint32_t tiles_per_run; ///< streamed tiles per run
    Where held;                  ///< the operand unpacked once per run (SrcA or SrcB), or None: every operand streams
};

/// Which FPU elementwise function an eltwise binary Plan runs (MathPlan::function). 0 stays "unused".
enum class BinaryFunction : std::uint32_t
{
    None, ///< no function
    Add,  ///< in0 + in1
    Sub,  ///< in0 - in1
    Mul   ///< in0 * in1
};

/// What math (the FPU or the SFPU) does.
struct MathPlan
{
    Exu unit;                    ///< the unit doing the math: Fpu or Sfpu; Unpacker for a copy straight to Dest (math only waits)
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value
    bool dest_fp32;              ///< 32-bit Dest mode
    DataFormat format;           ///< the math (Src) format
    bool transpose_faces;        ///< write the faces in transposed order
    std::uint32_t function;      ///< op-specific: which eltwise binary or SFPU function; 0 when unused
    bool fast = false;           ///< SFPU functions: approximate mode
};

/// What the packer does.
struct PackPlan
{
    DataFormat reads;         ///< the Dest format
    DataFormat writes;        ///< the format in L1
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tile_bytes; ///< one tile in L1, in bytes
    std::uint32_t tiles;      ///< tiles to pack
    bool relu;                ///< clamp negatives to 0 on the way out
    Toggle l1_acc;            ///< add into what is in L1 instead of overwriting it (Runtime: the pack call's argument)
    bool untilize;            ///< write rows instead of tiles
    DataFormat in_format;     ///< the packer's input format (In_data_format): the format the packer converts from
    bool dest_fp32;           ///< Dest holds 32-bit datums
};

/// What every unit does for one Compute API call.
struct Plan
{
    Op op;           ///< which op produced this Plan; nothing branches on it
    UnpackPlan in0;  ///< the first operand's stream
    UnpackPlan in1;  ///< the second operand's stream
    UnpackWalk walk; ///< the loop nest over the streams
    MathPlan math;   ///< math's part
    PackPlan pack;   ///< the packer's part
    Work work;       ///< loop counts and tile geometry
};

} // namespace llk2
