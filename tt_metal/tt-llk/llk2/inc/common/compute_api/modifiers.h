// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One named modifier struct per op, passed as a template argument on every call. Every field has a default, so a
// call names only what it changes: matmul<matmul_modifiers{.transpose_in1 = true}>(x, w, acc).
// Class-type template arguments in C++17 rely on the device compiler's -ftt-nttp.

#pragma once

namespace llk2
{

struct matmul_modifiers
{
    bool transpose_in1 = false; // transpose the faces of in1 (today's IN1_TRANSPOSE_TILE / UNPACK_TRANSPOSE_FACES)
};

} // namespace llk2
