// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One named modifier struct per op, passed as a template argument on every call. Every field has a default, so a
// call names only what it changes: matmul<matmul_modifiers{.transpose_in1 = true}>(x, w, acc).
// Class-type template arguments in C++17 rely on the device compiler's -ftt-nttp.

#pragma once

#include "resolver/plan.h"   // Broadcast
#include "tensor/tensor.h"   // Thread

namespace llk2
{

struct matmul_modifiers
{
    bool transpose_in1 = false; // transpose the faces of in1 (today's IN1_TRANSPOSE_TILE / UNPACK_TRANSPOSE_FACES)
};

struct add_modifiers
{
    Broadcast bcast = Broadcast::None; // how in1 is spread over in0: Row is today's add_tiles_bcast_rows
};

struct sub_modifiers
{
    Broadcast bcast = Broadcast::None; // how in1 is spread over in0: Row is today's sub_tiles_bcast<BroadcastType::ROW>
};

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

} // namespace llk2
