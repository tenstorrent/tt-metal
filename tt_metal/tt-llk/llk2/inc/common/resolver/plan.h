// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the Resolver returns for one Compute API call (tt-llk-lab llk-tensor, report 3, section 3). Three results,
// one per Resolver step:
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

// Where an operand goes, or where math reads it from.
enum class Where : std::uint8_t
{
    None,
    SrcA,
    SrcB,
    Dest
};

// The Compute API op a Plan belongs to. Only the math and pack sides branch on it; unpack does not.
enum class Op : std::uint8_t
{
    None,
    Matmul,
    EltwiseBinary,
    Copy,
    Vec,
    Pack
};

// How an operand's tile is read when it is smaller than the other operand along an axis.
enum class Broadcast : std::uint8_t
{
    None,
    Row,   // one row repeated down every row
    Col,   // one column repeated across every column
    Scalar // one datum everywhere
};

// How a tensor sits in L1.
enum class Layout : std::uint8_t
{
    Tilized, // tiles, each tile face by face
    RowMajor // plain rows, tilized by the unpacker on the way in
};

struct Rep // an operand where math reads it
{
    Where where;
    DataFormat format; // the register format the unpacker writes
};

struct OpConfig // negotiation's result for one op
{
    Exu unit;
    Rep in0, in1;
    std::uint32_t math_fidelity;
    TensorSpec<6> result; // what it produces
};

struct Work // work allocation's result for one op
{
    std::uint32_t rt_dim; // output tiles along y
    std::uint32_t ct_dim; // output tiles along x
    std::uint32_t kt_dim; // input tiles folded into each output tile (matmul's K; 1 otherwise)
    std::uint32_t face_r_dim, num_faces;
};

// One operand as an unpacker reads it: formats, geometry, and how its L1 address moves through the walk.
// Address steps are in tiles of this operand.
struct UnpackPlan
{
    Where to;          // SrcA (unpacker 0), SrcB (unpacker 1), Dest (unpacker 0, to Dest), or None: not unpacked
    DataFormat reads;  // the format in L1
    DataFormat writes; // the register format
    std::uint32_t face_r_dim, num_faces;
    std::uint32_t tile_bytes; // one tile in L1
    Layout layout;
    Broadcast broadcast;
    bool transpose_faces;
    std::uint32_t tile_step; // tiles the address moves after each streamed tile (inside a run)
    std::uint32_t run_step;  // tiles the address moves from one run to the next
    std::uint32_t step_step; // tiles the address moves from one step to the next
};

// The order the unpackers walk their operands, for every op:
//   for each of `steps` steps:                    (matmul: K steps)
//     for each of `runs` runs:                    (matmul: one per reused tile)
//       unpack the held operand once, if any      (matmul: the reused tile)
//       repeat tiles_per_run times:               (one MOP iteration each)
//         unpack every streamed operand, then move its address by tile_step
struct UnpackWalk
{
    std::uint32_t steps;
    std::uint32_t runs;
    std::uint32_t tiles_per_run;
    Where held; // the operand unpacked once per run (SrcA or SrcB), or None: every operand streams
};

// Which FPU elementwise function an eltwise binary Plan runs (MathPlan::function). 0 stays "unused".
enum class BinaryFunction : std::uint32_t
{
    None,
    Add, // in0 + in1
    Sub, // in0 - in1
    Mul  // in0 * in1
};

struct MathPlan
{
    Exu unit;
    std::uint32_t math_fidelity;
    bool dest_fp32;
    DataFormat format; // the math (Src) format
    bool transpose_faces;
    std::uint32_t function; // op-specific: which eltwise binary or SFPU function; 0 when unused
    bool fast = false;      // SFPU functions: approximate mode
};

struct PackPlan
{
    DataFormat reads;  // the Dest format
    DataFormat writes; // the format in L1
    std::uint32_t face_r_dim, num_faces;
    std::uint32_t tile_bytes; // one tile in L1
    std::uint32_t tiles;
    bool relu;      // clamp negatives to 0 on the way out
    bool l1_acc;    // add into what is in L1 instead of overwriting it
    bool untilize;  // write rows instead of tiles
    DataFormat in_format; // the packer's input format (In_data_format): the format the packer converts from
    bool dest_fp32;       // Dest holds 32-bit datums
};

struct Plan
{
    Op op;
    UnpackPlan in0, in1;
    UnpackWalk walk;
    MathPlan math;
    PackPlan pack;
    Work work;
};

} // namespace llk2
