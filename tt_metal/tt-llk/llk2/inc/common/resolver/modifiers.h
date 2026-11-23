// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The modifier structs: what a call asks of an op beyond its tensors. They sit beside the layers, next to the Plan
// (resolver/plan.h): a kernel writes one in a Compute API call, the Compute API hands it to the Resolver, and
// the Resolver's accepts / options / plan read it. Arch-independent.
//
// Modifiers. One named modifier struct per op (one per family), passed as a template argument on every call. Every
// field has a default, so a call names only what it changes: matmul<matmul_modifiers{.transpose_in1 = true}>(in0, in1,
// acc). A modifier the hardware has but the stack does not do yet is still a field, rejected by the Resolver's accepts
// as "not ported" (pack's untilize, reduce's dim, .accumulate for add and sub, a matmul that overwrites).
//
// One name for one thing across the ops. .accumulate: an FPU op folds its result into what acc holds in Dest instead
// of overwriting it (mul adds, reduce takes the max or adds; matmul has only this; add and sub do not have it yet).
// .l1_acc is the packer's: it adds into what L1 holds.
// Class-type template arguments in C++17 rely on the device compiler's -ftt-nttp.
//
// A modifier that is a single config bit is a Toggle (resolver/plan.h): Off, On or Runtime.
//   Off, On   fixed at compile time; the call takes no extra argument and the bit is written as an immediate
//   Runtime   the call takes the value as a run-time argument (after its tensors), and the bit is written from it;
//             one build serves both values. Compile-time checks treat the feature as possibly on, so a Runtime
//             modifier is accepted only where On would be.
//   pack<pack_modifiers{.l1_acc = Toggle::Runtime}>(acc, out, /* l1_acc */ block > 0);

#pragma once

#include <cstdint>

#include "resolver/plan.h"            // Broadcast, Toggle
#include "resolver/reduce_function.h" // ReduceKind, ReduceDim
#include "resolver/vec_function.h"    // VecFunction
#include "tensor/tensor.h"            // Thread

namespace llk2
{

/// Modifiers of matmul.
struct matmul_modifiers
{
    bool transpose_in1 = false; ///< use each in1 tile transposed, the tile grid unchanged (LLK 1.0's matmul transpose, a product with K^T)
    bool accumulate    = true;  ///< add the products into acc (y = acc + in0 * in1): what the FPU's MVMUL does, and a Dest ring
                                ///< section starts at zero. Overwriting acc (false) is not ported, rejected at compile time
};

/// Modifiers of add, sub and mul: one family, one struct.
struct eltwise_binary_modifiers
{
    Broadcast bcast = Broadcast::None; ///< how in1 is spread over in0: Row is LLK 1.0's add_tiles_bcast_rows /
                                       ///< sub_tiles_bcast<BroadcastType::ROW>, Col LLK 1.0's mul_tiles_bcast_cols
    bool accumulate = false;           ///< add the result into acc (y = acc + in0 * in1) instead of overwriting acc. mul only:
                                       ///< not ported for add and sub, rejected at compile time
};

using add_modifiers = eltwise_binary_modifiers; ///< Modifiers of add.
using sub_modifiers = eltwise_binary_modifiers; ///< Modifiers of sub.
using mul_modifiers = eltwise_binary_modifiers; ///< Modifiers of mul.

/// Modifiers of reduce.
struct reduce_modifiers
{
    ReduceKind kind = ReduceKind::Max; ///< Max (LLK 1.0's PoolType::MAX) or Sum (PoolType::SUM)
    ReduceDim dim   = ReduceDim::Row;  ///< Row: over each row's columns, into column 0 (LLK 1.0's REDUCE_ROW); the only one ported
    bool accumulate = false;           ///< fold the reduction into what acc holds instead of overwriting it (y = max(acc, row max),
                                       ///< y = acc + row sum). acc then holds a seed where the reduction folds: compute_api/reduce.h
};

/// Modifiers of copy.
struct copy_modifiers
{
    bool transpose = false; ///< width-height transpose of each tile on the way to Dest (LLK 1.0's transpose_wh_tile)
};

/// Modifiers of every SFPU function (vec_silu, vec_tanh, vec_exp, ...): one family, one struct. What a function does
/// not have (a form, a region, a scale) is refused by the Resolver's accepts.
struct vec_modifiers
{
    VecForm form = VecForm::Exact;            ///< which of the function's algorithms: Exact, Fast (the functions that have one) or
                                              ///< Polynomial (exp on LeftFacesEvenColumns)
    VecRegion region = VecRegion::All;        ///< the datums of each tile the function rewrites: All, LeftFaces or LeftFacesEvenColumns
    bool clamp_input = true;                  ///< the result is right for every input. false: the caller keeps the input in the form's
                                              ///< range. The fast exp has only that: Schraudolph's approximation without input clamping,
                                              ///< right for input_scale x in [-88, 0.72] and negative below (LLK 1.0's InputClamping::None)
    Thread thread = Thread::Math;             ///< the thread that drives the SFPU: Thread::Math or Thread::Pack (the target kernel's choice).
                                              ///< One thread for all of a kernel's vec_ calls that may overlap (compute_api/vec.h)
    std::uint32_t input_scale = vec_fp32_one; ///< exp only: y = exp(input_scale x), an FP32 bit pattern known at compile time (the
                                              ///< fast exp folds it into its constants; the other forms round it to bf16)
};

/// What a vec_ call hands the Resolver: the function it names and the caller's modifiers.
struct vec_call
{
    VecFunction function; ///< which function
    vec_modifiers mods;   ///< the caller's modifiers
};

/// Modifiers of pack.
struct pack_modifiers
{
    bool relu      = false;       ///< ReLU on the way out: negatives become 0 (the packer's ZERO_RELU mode)
    Toggle l1_acc  = Toggle::Off; ///< add into what is already in L1 instead of overwriting it (the packer's L1 accumulation, Pack_L1_Acc)
    bool untilize  = false;       ///< write rows instead of tiles; not ported yet, rejected at compile time
    bool sum_tiles = false;       ///< dst is one tile: every tile of src is added up into it, in dst's format in L1 (the packer's
                                  ///< L1 accumulation from the second tile on). l1_acc then says what the first tile does:
                                  ///< Off overwrites dst (dst = the sum), On adds as well (dst += the sum)
};

} // namespace llk2
