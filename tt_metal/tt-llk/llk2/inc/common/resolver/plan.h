// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the Resolver returns for one Compute API call (report 3, section 3), cut down to what matmul needs:
//   OpConfig  negotiation: where each operand sits in registers and in which format, and the result's spec
//   Work      work allocation: loop counts
//   Plan      planning: what each unit does
// The LLK API turns a Plan into its op's named parameter struct (matmul_params, ...).

#pragma once

#include <cstdint>

#include "tensor/tensor.h"

namespace llk2
{

enum class Where : std::uint8_t
{
    None,
    SrcA,
    SrcB,
    Dest
};

enum class MathOp : std::uint8_t
{
    None,
    Matmul
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
    TensorSpec<6> result; // what it produces: in Dest for math
};

struct Work // work allocation's result for one op
{
    std::uint32_t rt_dim; // output tiles along y
    std::uint32_t ct_dim; // output tiles along x
    std::uint32_t kt_dim; // input tiles folded into each output tile
    std::uint32_t face_r_dim, num_faces;
};

struct UnpackPlan // one operand
{
    Where to;
    DataFormat reads;  // the format in L1
    DataFormat writes; // the register format
    std::uint32_t face_r_dim, num_faces;
    std::uint32_t tile_bytes; // how far it moves in L1 from one tile to the next
    bool transpose_faces;
};

struct MathPlan
{
    Exu unit;
    MathOp op;
    std::uint32_t math_fidelity;
    bool dest_fp32;
    DataFormat format; // the math (Src) format
    bool transpose_faces;
};

struct Plan
{
    UnpackPlan in0, in1;
    MathPlan math;
    Work work;
};

} // namespace llk2
