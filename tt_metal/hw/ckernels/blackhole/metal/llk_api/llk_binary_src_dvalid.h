// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_operands.h"

// A row or column broadcast takes the per-tile hand-off only from a full 32x32 B tile, since its SrcB layout reads B as
// 16-row faces. The unpack and math inits and the math call all fall back to per face through these two definitions.
template <ckernel::BroadcastType src_b_bcast_type, ckernel::SrcDvalid src_dvalid>
inline constexpr bool eltwise_binary_bcast_per_tile =
    src_dvalid == ckernel::SrcDvalid::PerTile &&
    (src_b_bcast_type == ckernel::BroadcastType::COL || src_b_bcast_type == ckernel::BroadcastType::ROW);

// A macro rather than a bool function, so each site branches on the test itself; GCC compiles the two differently.
#define ELTWISE_BINARY_B_TILE_NOT_FULL(operand_b_id) \
    (get_operand_face_r_dim(operand_b_id) != ckernel::FACE_R_DIM || get_operand_num_faces(operand_b_id) != 4)
