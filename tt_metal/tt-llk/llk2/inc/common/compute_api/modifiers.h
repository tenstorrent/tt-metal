// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One named modifier struct per op, passed as a template argument on every call. Every field has a default, so a
// call names only what it changes: matmul<matmul_modifiers{.transpose_in1 = true}>(in0, in1, acc).
// Class-type template arguments in C++17 rely on the device compiler's -ftt-nttp.
//
// A modifier that is a single config bit is a Toggle (resolver/plan.h): Off, On or Runtime.
//   Off, On   fixed at compile time; the call takes no extra argument and the bit is written as an immediate
//   Runtime   the call takes the value as a run-time argument (after its tensors), and the bit is written from it;
//             one build serves both values. Compile-time checks treat the feature as possibly on, so a Runtime
//             modifier is accepted only where On would be.
//   pack<pack_modifiers{.l1_acc = Toggle::Runtime}>(acc, out, /* l1_acc */ block > 0);

#pragma once

#include "resolver/plan.h" // Broadcast
#include "tensor/tensor.h" // Thread

namespace llk2
{

/// Modifiers of matmul.
struct matmul_modifiers
{
    bool transpose_in1 = false; ///< transpose the faces of in1 (today's IN1_TRANSPOSE_TILE / UNPACK_TRANSPOSE_FACES); not ported yet, rejected at compile time
};

/// Modifiers of add.
struct add_modifiers
{
    Broadcast bcast = Broadcast::None; ///< how in1 is spread over in0: Row is today's add_tiles_bcast_rows
};

/// Modifiers of sub.
struct sub_modifiers
{
    Broadcast bcast = Broadcast::None; ///< how in1 is spread over in0: Row is today's sub_tiles_bcast<BroadcastType::ROW>
};

/// Modifiers of copy.
struct copy_modifiers
{
    bool transpose = false; ///< width-height transpose of each tile on the way to Dest (today's transpose_wh_tile)
};

/// Modifiers of every SFPU activation (vec_silu, vec_tanh, ...).
struct vec_modifiers
{
    bool fast     = false;        ///< approximate mode: today's fast_and_approx / APPROX template argument
    Thread thread = Thread::Math; ///< the thread that drives the SFPU: Thread::Math or Thread::Pack (the target kernel's choice)
};

/// Modifiers of pack.
struct pack_modifiers
{
    bool relu     = false;       ///< ReLU on the way out: negatives become 0 (the packer's ZERO_RELU mode)
    Toggle l1_acc = Toggle::Off; ///< add into what is already in L1 instead of overwriting it (the packer's L1 accumulation, Pack_L1_Acc)
    bool untilize = false;       ///< write rows instead of tiles; not ported yet, rejected at compile time
};

} // namespace llk2
