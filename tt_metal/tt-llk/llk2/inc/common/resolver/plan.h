// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the Resolver returns for one Compute API call (tt-llk-lab llk-tensor, report 3, section 3), cut down to
// what matmul needs. Three results, one per Resolver step:
//   OpConfig  options (negotiation): which unit does the math, the register format of each operand, the fidelity,
//             and the spec of the result
//   Work      work (allocation): loop counts, from the tensors once the config is fixed
//   Plan      plan: what each unit does. llk-lib turns it into each thread's parameter struct
//             (unpack_params_from, matmul_params_from)

#pragma once

#include <cstdint>

#include "tensor/tensor.h"

namespace llk2
{

// Where math reads an operand from.
enum class Where : std::uint8_t
{
    None,
    SrcA,
    SrcB,
    Dest
};

// The math operation a Compute API call performs. llk-lib's generic per-thread calls (unpack, ...) pick the parts
// of their implementation that differ per op from this.
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
    std::uint32_t math_fidelity;
    bool dest_fp32;
    DataFormat format; // the math (Src) format
    bool transpose_faces;
};

struct Plan
{
    MathOp op;
    UnpackPlan in0, in1;
    MathPlan math;
    Work work;
};

} // namespace llk2
